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
-- Automatically adds a Description section based on frontmatter using _render_meta.py
-- Also processes SQL code blocks using _render_examples.py

local function render_sql_example(sql_code)
  -- Create a temporary file for the SQL
  local temp_file = os.tmpname()

  -- Write SQL content to temporary file
  local temp_handle = io.open(temp_file, "w")
  if not temp_handle then
    return {pandoc.CodeBlock(sql_code, pandoc.Attr("", {"sql"}, {}))}
  end

  temp_handle:write(sql_code)
  temp_handle:close()

  -- Execute _render_examples.py using stdin (with - argument)
  local cmd = string.format("python3 _render_examples.py - < '%s'", temp_file)
  local handle = io.popen(cmd, "r")
  if not handle then
    os.remove(temp_file)
    return {pandoc.CodeBlock(sql_code, pandoc.Attr("", {"sql"}, {}))}
  end

  local result = handle:read("*all")
  handle:close()
  os.remove(temp_file)

  -- Parse the markdown result and return as pandoc blocks
  if result and result ~= "" then
    local doc_result = pandoc.read(result, "markdown")
    return doc_result.blocks
  end

  -- Fallback to original code block if rendering fails
  return {pandoc.CodeBlock(sql_code, pandoc.Attr("", {"sql"}, {}))}
end

local function render_meta_content(doc)
  -- Use Pandoc's JSON encoding (may not work perfectly)
  local json_content = pandoc.json.encode(doc.meta)

  -- Create a temporary file for the JSON
  local temp_file = os.tmpname()

  -- Write JSON content to temporary file
  local temp_handle = io.open(temp_file, "w")
  if not temp_handle then
    return pandoc.Para{pandoc.Str("Error: Could not create temporary file")}
  end

  temp_handle:write(json_content)
  temp_handle:close()

  -- Pass JSON directly to _render_meta.py (JSON is valid YAML)
  local cmd = string.format("python3 _render_meta.py - < '%s'", temp_file)

  local handle = io.popen(cmd, "r")
  if not handle then
    os.remove(temp_file)
    return pandoc.Para{pandoc.Str("Error: Could not execute _render_meta.py")}
  end

  local result = handle:read("*all")
  if not result then
    result = "Error: Could not read output from _render_meta.py"
  end

  handle:close()
  os.remove(temp_file)  -- Parse the markdown result and return as pandoc blocks
  if result and result ~= "" then
    local doc_result = pandoc.read(result, "markdown")
    return doc_result.blocks
  end

  return {}
end

function Pandoc(doc)
  local description = doc.meta.description

  if description and description ~= "" then
    -- First pass: Process all existing SQL code blocks before generating new content
    local new_blocks = {}
    for i, block in ipairs(doc.blocks) do
      if block.t == "CodeBlock" and block.classes and #block.classes > 0 and block.classes[1] == "sql" then
        local rendered_blocks = render_sql_example(block.text)
        -- Add all rendered blocks to new_blocks
        for _, rendered_block in ipairs(rendered_blocks) do
          table.insert(new_blocks, rendered_block)
        end
      else
        table.insert(new_blocks, block)
      end
    end
    doc.blocks = new_blocks

    -- Second pass: Generate content using _render_meta.py
    local meta_blocks = render_meta_content(doc)

    if #meta_blocks > 0 then
      -- Insert after title (first header) if it exists, otherwise at the beginning
      local insert_pos = 1
      for i, block in ipairs(doc.blocks) do
        if block.t == "Header" and block.level == 1 then
          insert_pos = i + 1
          break
        end
      end

      -- Insert the generated blocks
      for j, block in ipairs(meta_blocks) do
        table.insert(doc.blocks, insert_pos + j - 1, block)
      end
    end
  end

  return doc
end
