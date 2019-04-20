# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

import json
from graphviz import *

def json2graph(json_str):
    try:
        conf_dic = json.loads(json_str)
    except ValueError as e:
        return e, False
        # return "ERROR JSON!"

    color = {
        "Input": "royalblue",
        "Embedding": "orange",
        "Linear": "tan",
        "LinearAttention": "tan",
        "BiGRU": "salmon",
        "BiLSTM": "salmon",
        "BiLSTMAtt": "salmon1",
        "BiGRULast": "salmon",
        "Conv": "sandybrown",
        "ConvPooling": "sandybrown",
        "Pooling": "skyblue",
        "Dropout": "yellowgreen",
        "Combination": "purple",
        "EncoderDecoder": "lightsalmon",
        "FullAttention": "lightsalmon",
        "Seq2SeqAttention": "lightsalmon"
    }
    layer_conf = {
        "Linear": ["hidden_dim", "activation", "last_hidden_activation", "last_hidden_softmax", "batch_normalization"],
        "LinearAttention": ["keep_dim"],
        "BiGRU": ["hidden_dim", "dropout"],
        "BiGRULast": ["hidden_dim", "dropout"],
        "BiLSTM": ["hidden_dim", "dropout", "num_layers"],
        "BiLSTMAtt": ["hidden_dim", "dropout", "num_layers"],
        "Conv": ["stride", "padding", "window_sizes", "input_channel_num", "output_channel_num", "activation",
                 "batch_normalization"],
        "ConvPooling": ["stride", "padding", "window_sizes", "input_channel_num", "output_channel_num",
                        "batch_normalization",
                        "activation", "pool_type", "pool_axis"],
        "Pooling": ["pool_axis", "pool_type"],
        "Dropout": ["dropout"],
        "Combination": ["operations"],
        "EncoderDecoder": ["encoder", "decoder"],
        "FullAttention": ["hidden_dim", "activation"],
        "Seq2SeqAttention": ["attention_dropout"]
    }

    model = Digraph(format='svg',
                    node_attr={"style": "rounded, filled",
                               "shape": "box",
                               "fontcolor": "white"})
    model.attr(rankdir="BT")

    for item in conf_dic['architecture']:
        if item['layer'] == "Embedding":
            for c in item['conf']:
                dim = item['conf'][c]['dim']
                for n in item['conf'][c]['cols']:
                    label_str = "<" \
                                + "<table border='0.5' align='center'>" \
                                + "<tr><td align='text'><i>" + n + "</i></td>" + "<td align='text'><b>Embedding</b></td></tr>" \
                                + "<tr><td align='text'>dim:</td>" + "<td align='text'>" + str(dim) + "</td></tr>" \
                                + "</table>>"
                    model.node(name=n, label=label_str, fillcolor=color["Embedding"])
            break

    for inp in conf_dic['inputs']['model_inputs']:
        model.node(name=inp,
                   label=inp,
                   fillcolor=color['Input'])
        for n in conf_dic['inputs']['model_inputs'][inp]:
            model.edge(n, inp)

    layer_dic = {}
    for item in conf_dic['architecture']:
        if 'layer_id' in item.keys() and 'layer' in item.keys() and 'conf' in item.keys():
            layer_dic[item['layer_id']] = [item['layer'], item['conf']]

    for item in conf_dic['architecture']:
        if 'layer_id' in item.keys():
            if item['layer'] in layer_dic:
                tmp_layer = item['layer']
                item['conf'] = layer_dic[tmp_layer][1]
                item['layer'] = layer_dic[tmp_layer][0]
            label_str = "<" \
                        + "<table border='0.5' align='center'>" \
                        + "<tr><td align='text'><i>" + item['layer_id'] + "</i></td>" + "<td align='text'><b>" + item[
                            'layer'] + "</b></td></tr>"
            if item['layer'] in layer_conf:
                for c in layer_conf[item['layer']]:
                    if c in item['conf']:
                        label_str = label_str + "<tr><td align='text'>" + c + "</td>" + "<td align='text'>" + str(
                            item['conf'][c]) + "</td></tr>"
            else:
                for c in item['conf']:
                    label_str = label_str + "<tr><td align='text'>" + c + "</td>" + "<td align='text'>" + str(
                        item['conf'][c]) + "</td></tr>"

            label_str += "</table>>"

            model.node(name=item['layer_id'],
                       label=label_str,
                       fillcolor=color.get(item['layer'], "grey"))
            for inp in item['inputs']:
                model.edge(inp, item['layer_id'])
    # model
    model.render('static/graph.gv', view=False)

    return
