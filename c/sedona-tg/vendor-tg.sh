#!/usr/bin/env bash
#
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

TG_REF=e136401cd6870008eb7d12a2a165fdc27d18d5ef

curl -L https://github.com/tidwall/tg/raw/${TG_REF}/tg.c \
    -o src/tg/tg.c

curl -L https://github.com/tidwall/tg/raw/${TG_REF}/tg.h \
    -o src/tg/tg.h
