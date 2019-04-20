# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

# -*- coding:utf-8 -*-
import web
import json
from web import form
from mv import json2graph

render = web.template.render('templates/')
urls = (
    '/', 'index',
    '/mv', 'model_visualizer'
)

mv_form = form.Form(
    form.Textarea("json", description="config_json"),
    form.Textarea("output", description="output"),
    form.Button("submit", type="submit", description="Submit"),
)


class index:
    def GET(self):
        raise web.seeother('/mv')


class model_visualizer:
    def GET(self):
        f = mv_form()
        status = False
        return render.model_visualizer(f, status)


    def POST(self):
        f = mv_form()
        post_value = web.input(json=None)
        f['json'].value = post_value.json
        json2graph(post_value.json)
        status = True
        return render.model_visualizer(f, status)

if __name__ == "__main__":
    app = web.application(urls, globals())
    app.run()




