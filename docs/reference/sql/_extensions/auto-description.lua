-- Licensed to the Apache Software Foundation (ASF) under one
-- or more contributor license agreements.  See the NOTICE file
-- distributed with this work for additional information
-- regarding copyright ownership.  The ASF licenses this file
-- to you under the Apache License, Version 2.0 (the
-- "License"); you may not use this file except in compliance
-- with the License.  You may obtain a copy of the License at
--
--   http://www.apache.org/licenses/LICENSE-2.0
--
-- Unless required by applicable law or agreed to in writing,
-- software distributed under the License is distributed on an
-- "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
-- KIND, either express or implied.  See the License for the
-- specific language governing permissions and limitations
-- under the License.

-- auto-description.lua
-- Automatically adds a Description section based on frontmatter description

function Pandoc(doc)
  local description = doc.meta.description

  if description and description ~= "" then
    local description_header = pandoc.Header(2, {pandoc.Str("Description")})
    local description_content = pandoc.Para({
      pandoc.RawInline('html', '{{< meta description >}}')
    })

      -- Insert after title (first header) if it exists, otherwise at the beginning
    local insert_pos = 1
    for i, block in ipairs(doc.blocks) do
      if block.t == "Header" and block.level == 1 then
        insert_pos = i + 1
        break
      end
    end

    table.insert(doc.blocks, insert_pos, description_header)
    table.insert(doc.blocks, insert_pos + 1, description_content)
    table.insert(doc.blocks, insert_pos + 2, pandoc.Para({})) -- Empty line
  end

  return doc
end
