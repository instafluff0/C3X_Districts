"""Adapt normalized authored era weights/order to stable Lab building choices.

This consumes source parameters; it does not reproduce the source placement
algorithm, hex topology, population interpolation, or road/spine generation.
"""
import math


def era_sequence(layers, count):
    if not layers or count < 1:
        raise ValueError('city generator needs layers and slots')
    layers = sorted(layers, key=lambda x: x['order_from_center'])
    if any(not math.isfinite(x['weight']) or x['weight'] <= 0 for x in layers):
        raise ValueError('city layer weights must be positive and finite')
    total = sum(x['weight'] for x in layers)
    used = [0] * len(layers)
    result = []
    for i in range(count):
        # Reserve the first source compound for the central art era. Subsequent
        # deficit selection approximates weights while preserving every prefix.
        index = 0 if i == 0 else max(range(len(layers)), key=lambda j:
            (layers[j]['weight'] * (i + 1) / total - used[j], -layers[j]['order_from_center']))
        used[index] += 1
        result.append(layers[index])
    return result


def select_components(layers, components, compound_ids, count=11):
    result = []
    indices = {x['era']: 0 for x in layers}
    for slot, layer in enumerate(era_sequence(layers, count)):
        era = layer['era']
        assets = components[era]
        compounds = sorted([x for x in assets if x['id'] in compound_ids],
                           key=lambda x: (-(x['hi'][2] - x['lo'][2]), x['id']))
        houses = sorted([x for x in assets if x['id'] not in compound_ids],
                        key=lambda x: (-(x['hi'][2] - x['lo'][2]), x['id']))
        if not houses:
            raise ValueError('city layer has no standalone source buildings: ' + era)
        if slot == 0 and compounds:
            asset = compounds[0]
        else:
            asset = houses[indices[era] % len(houses)]
            indices[era] += 1
        result.append({**asset, 'era_layer': era,
                       'order_from_center': layer['order_from_center'],
                       **({'layout_radius': layer['order_from_center'] * .24} if len(layers)>1 else {})})
    return result
