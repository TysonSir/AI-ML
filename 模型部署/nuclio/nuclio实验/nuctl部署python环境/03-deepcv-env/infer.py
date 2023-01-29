# Copyright 2017 The Nuclio Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch

def handler(context, event):
    context.logger.info('This is an unstructured log')

    return context.Response(body='Hello, from nuclio :] czz deepcv2, torch_version: ' + torch.__version__,
                            headers={},
                            content_type='text/plain',
                            status_code=200)

'''
安装：
nuctl deploy --path /home/server/czz-nuclio-test/03-deepcv-env

卸载：
docker stop nuclio-nuclio-infer-func && docker rm nuclio-nuclio-infer-func && docker rmi nuclio/processor-infer-func
'''