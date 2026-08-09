"""Assisted authoring and diagnostic checks for evaluation campaigns."""

from __future__ import annotations

import os
import re
import unicodedata
from dataclasses import dataclass
from typing import Any

from semantic_evaluation.core.campaign_designer import CampaignWorkspace


@dataclass(frozen=True)
class CampaignIssue:
    """One actionable problem found in a campaign workspace."""

    severity: str
    code: str
    message: str
    reference: str = ''


def identifier_from_query(text: str, fallback: str = 'query') -> str:
    """Create a stable, YAML-friendly identifier from natural language."""
    ascii_text = unicodedata.normalize('NFKD', text.casefold()).encode(
        'ascii', 'ignore'
    ).decode('ascii')
    normalized = re.sub(r'[^a-z0-9]+', '_', ascii_text).strip('_')
    return normalized[:48] or fallback


def unique_case_id(workspace: CampaignWorkspace, requested: str) -> str:
    """Return ``requested`` or the first unused numeric variant."""
    base = identifier_from_query(requested, 'query')
    used = {
        str(case.get('case_id', '')).strip()
        for case in workspace.queries.get('cases', [])
        if isinstance(case, dict)
    }
    if base not in used:
        return base
    suffix = 2
    while f'{base}_{suffix}' in used:
        suffix += 1
    return f'{base}_{suffix}'


def upsert_query_case(
    workspace: CampaignWorkspace,
    *,
    case_id: str,
    query_id: str,
    query_text: str,
    query_type: str,
    language: str,
    start_pose_id: str,
    exact_valid_nodes: list[str],
    nearby_valid_nodes: list[str],
    is_negative: bool,
    target_visible: bool,
    timeout_s: float,
) -> dict[str, Any]:
    """Insert or replace a confirmed query and synchronize negative GT."""
    normalized_case_id = case_id.strip()
    normalized_query_id = query_id.strip()
    normalized_text = query_text.strip()
    if not normalized_case_id or not normalized_query_id or not normalized_text:
        raise ValueError('case_id, query_id and query_text are required')
    if timeout_s <= 0.0:
        raise ValueError('timeout_s must be positive')
    node_ids = {node.node_id for node in workspace.nodes}
    exact = _unique(exact_valid_nodes)
    nearby = [value for value in _unique(nearby_valid_nodes) if value not in exact]
    unknown = sorted((set(exact) | set(nearby)) - node_ids)
    if unknown:
        raise ValueError(f'query references unknown nodes: {unknown}')
    if is_negative:
        exact = []
        nearby = []
        target_visible = False
    elif not exact:
        raise ValueError('a positive query requires at least one exact node')

    cases = workspace.queries.setdefault('cases', [])
    existing = next(
        (
            case for case in cases
            if isinstance(case, dict)
            and str(case.get('case_id', '')).strip() == normalized_case_id
        ),
        None,
    )
    if existing is None:
        existing = {}
        cases.append(existing)
    previous_query_id = str(existing.get('query_id', '')).strip()
    existing.clear()
    existing.update({
        'case_id': normalized_case_id,
        'query_id': normalized_query_id,
        'query_text': normalized_text,
        'query_type': query_type.strip() or ('negative' if is_negative else 'object'),
        'language': language.strip() or 'es',
        'start_pose_id': start_pose_id.strip() or 'origin',
        'exact_valid_nodes': exact,
        'nearby_valid_nodes': nearby,
        'is_negative': bool(is_negative),
        'target_visible': bool(target_visible),
        'timeout_s': float(timeout_s),
    })
    negative_queries = workspace.ground_truth.setdefault('negative_queries', [])
    for identifier in (previous_query_id, normalized_query_id):
        while identifier and identifier in negative_queries:
            negative_queries.remove(identifier)
    if is_negative:
        negative_queries.append(normalized_query_id)
    return existing


def confirm_object_ground_truth(
    workspace: CampaignWorkspace,
    evidence_by_node: dict[str, list[str]],
    *,
    exact_nodes: set[str],
    nearby_nodes: set[str],
) -> int:
    """Confirm explicitly reviewed object labels for annotated nodes."""
    nodes = {node.node_id: node for node in workspace.nodes}
    entries = workspace.ground_truth.setdefault('objects', [])
    changed = 0
    for node_id, labels in evidence_by_node.items():
        node = nodes.get(node_id)
        if node is None:
            continue
        room_id = node.room_id or ''
        for category in _unique(labels):
            entry = next(
                (
                    candidate for candidate in entries
                    if isinstance(candidate, dict)
                    and str(candidate.get('category', '')) == category
                    and str(candidate.get('room_id', '')) == room_id
                ),
                None,
            )
            if entry is None:
                entry = {
                    'category': category,
                    'room_id': room_id,
                    'exact_valid_nodes': [],
                    'nearby_valid_nodes': [],
                }
                entries.append(entry)
            key = (
                'exact_valid_nodes' if node_id in exact_nodes
                else 'nearby_valid_nodes' if node_id in nearby_nodes
                else ''
            )
            if key and node_id not in entry.setdefault(key, []):
                entry[key].append(node_id)
                changed += 1
    return changed


def campaign_issues(workspace: CampaignWorkspace) -> list[CampaignIssue]:
    """Return errors and warnings without changing any annotations."""
    issues: list[CampaignIssue] = []
    node_ids = {node.node_id for node in workspace.nodes}
    room_ids = {room.room_id for room in workspace.rooms}
    if not node_ids:
        issues.append(CampaignIssue(
            'error', 'empty_graph', 'El grafo no contiene nodos semánticos.'
        ))
    cases = workspace.queries.get('cases', [])
    if not isinstance(cases, list):
        return [CampaignIssue(
            'error', 'invalid_cases', 'queries.cases debe ser una lista.'
        )]
    if not cases:
        issues.append(CampaignIssue(
            'warning', 'empty_suite', 'La campaña no contiene consultas.'
        ))

    seen_cases: set[str] = set()
    seen_queries: set[str] = set()
    seen_texts: dict[str, str] = {}
    for index, case in enumerate(cases):
        reference = f'cases[{index}]'
        if not isinstance(case, dict):
            issues.append(CampaignIssue(
                'error', 'invalid_case', 'El caso debe ser un mapping.', reference
            ))
            continue
        case_id = str(case.get('case_id', '')).strip()
        query_id = str(case.get('query_id', '')).strip()
        text = str(case.get('query_text', '')).strip()
        reference = case_id or reference
        for field, value in (
            ('case_id', case_id), ('query_id', query_id), ('query_text', text)
        ):
            if not value:
                issues.append(CampaignIssue(
                    'error', f'missing_{field}', f'Falta {field}.', reference
                ))
        if case_id in seen_cases:
            issues.append(CampaignIssue(
                'error', 'duplicate_case_id', f'case_id duplicado: {case_id}', reference
            ))
        seen_cases.add(case_id)
        if query_id in seen_queries:
            issues.append(CampaignIssue(
                'error', 'duplicate_query_id', f'query_id duplicado: {query_id}', reference
            ))
        seen_queries.add(query_id)
        normalized_text = ' '.join(text.casefold().split())
        if normalized_text and normalized_text in seen_texts:
            issues.append(CampaignIssue(
                'warning', 'duplicate_text',
                f'Mismo texto que {seen_texts[normalized_text]}.', reference,
            ))
        elif normalized_text:
            seen_texts[normalized_text] = reference

        exact = _string_list(case.get('exact_valid_nodes'))
        nearby = _string_list(case.get('nearby_valid_nodes'))
        unknown = sorted((set(exact) | set(nearby)) - node_ids)
        if unknown:
            issues.append(CampaignIssue(
                'error', 'unknown_query_nodes',
                f'Nodos de criterio inexistentes: {", ".join(unknown)}.', reference,
            ))
        overlap = sorted(set(exact) & set(nearby))
        if overlap:
            issues.append(CampaignIssue(
                'error', 'overlapping_validity',
                f'Nodos simultáneamente exactos y cercanos: {", ".join(overlap)}.',
                reference,
            ))
        negative = bool(case.get('is_negative', False))
        visible = bool(case.get('target_visible', not negative))
        if negative and (exact or nearby or visible):
            issues.append(CampaignIssue(
                'error', 'invalid_negative_query',
                'Una consulta negativa no puede tener nodos válidos ni objetivo visible.',
                reference,
            ))
        if not negative and not exact:
            issues.append(CampaignIssue(
                'error', 'positive_without_exact',
                'La consulta positiva no tiene nodos exactos.', reference,
            ))
        try:
            timeout = float(case.get('timeout_s', 0.0) or 0.0)
        except (TypeError, ValueError):
            timeout = 0.0
        if timeout <= 0.0:
            issues.append(CampaignIssue(
                'error', 'invalid_timeout', 'El timeout debe ser positivo.', reference
            ))

    gt_negative = set(_string_list(
        workspace.ground_truth.get('negative_queries')
    ))
    declared_negative = {
        str(case.get('query_id', '')).strip()
        for case in cases
        if isinstance(case, dict) and bool(case.get('is_negative', False))
    }
    if gt_negative != declared_negative:
        issues.append(CampaignIssue(
            'error', 'negative_gt_mismatch',
            'negative_queries no coincide con las consultas negativas de la suite.',
        ))

    detected_labels = {
        detected.label
        for node in workspace.nodes
        for observation in node.observations
        for detected in observation.objects
    }
    for group in ('rooms', 'objects', 'relations'):
        entries = workspace.ground_truth.get(group, [])
        if not isinstance(entries, list):
            issues.append(CampaignIssue(
                'error', f'invalid_gt_{group}',
                f'ground_truth.{group} debe ser una lista.', group,
            ))
            continue
        for index, entry in enumerate(entries):
            reference = f'ground_truth.{group}[{index}]'
            if not isinstance(entry, dict):
                issues.append(CampaignIssue(
                    'error', 'invalid_gt_entry',
                    'La anotación debe ser un mapping.', reference,
                ))
                continue
            keys = (
                ('exact_valid_nodes', 'nearby_valid_nodes')
                if group == 'objects' else ('valid_nodes',)
            )
            referenced = {
                node_id
                for key in keys
                for node_id in _string_list(entry.get(key))
            }
            unknown = sorted(referenced - node_ids)
            if unknown:
                issues.append(CampaignIssue(
                    'error', 'unknown_ground_truth_nodes',
                    f'Nodos GT inexistentes: {", ".join(unknown)}.', reference,
                ))
            if group == 'rooms':
                room_id = str(entry.get('room_id', '')).strip()
                if not room_id:
                    issues.append(CampaignIssue(
                        'error', 'missing_gt_room', 'Falta room_id.', reference,
                    ))
                elif room_ids and room_id not in room_ids:
                    issues.append(CampaignIssue(
                        'warning', 'gt_room_without_polygon',
                        f'La room GT {room_id} no tiene polígono.', reference,
                    ))
            elif group == 'objects':
                category = str(entry.get('category', '')).strip()
                if not category:
                    issues.append(CampaignIssue(
                        'error', 'missing_gt_category',
                        'Falta la categoría del objeto.', reference,
                    ))
                elif category not in detected_labels:
                    issues.append(CampaignIssue(
                        'warning', 'undetected_gt_category',
                        f'La categoría GT {category} no aparece detectada.', reference,
                    ))
            else:
                required = ('subject', 'predicate', 'object')
                missing = [key for key in required if not str(entry.get(key, '')).strip()]
                if missing:
                    issues.append(CampaignIssue(
                        'error', 'incomplete_gt_relation',
                        f'Faltan campos de relación: {", ".join(missing)}.', reference,
                    ))

    for node in workspace.nodes:
        if not node.room_id:
            issues.append(CampaignIssue(
                'warning', 'node_without_room',
                'El nodo no tiene room asignada.', node.node_id,
            ))
        elif room_ids and node.room_id not in room_ids:
            issues.append(CampaignIssue(
                'warning', 'unknown_node_room',
                f'La room {node.room_id} no tiene polígono cargado.', node.node_id,
            ))
        if not node.observations:
            issues.append(CampaignIssue(
                'warning', 'node_without_observations',
                'El nodo no tiene observaciones.', node.node_id,
            ))
        for observation in node.observations:
            reference = f'{node.node_id}/{observation.observation_id}'
            image_path = os.path.expanduser(observation.image_path)
            if not image_path or not os.path.isfile(image_path):
                issues.append(CampaignIssue(
                    'warning', 'missing_observation_image',
                    'La imagen de la observación no está disponible.', reference,
                ))
            if observation.transition_zone:
                issues.append(CampaignIssue(
                    'warning', 'transition_observation',
                    'Observación tomada en zona de transición.', reference,
                ))
            if observation.contamination_class == 'contaminated':
                issues.append(CampaignIssue(
                    'warning', 'contaminated_observation',
                    'Observación clasificada como contaminada.', reference,
                ))
    return issues


def _unique(values: list[str]) -> list[str]:
    output: list[str] = []
    for value in values:
        normalized = str(value).strip()
        if normalized and normalized not in output:
            output.append(normalized)
    return output


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, (list, tuple, set)):
        return []
    return [str(item).strip() for item in value if str(item).strip()]
