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

return {
  ["function-listing"] = function(args, kwargs, meta)
    -- Get the file pattern from arguments (default to "st_*.qmd")
    local file_pattern = args[1] or "st_*.qmd"

  -- Execute _render_listing.py with the file pattern
  local cmd = string.format("python3 _render_listing.py '%s'", file_pattern)
  local handle = io.popen(cmd, "r")

  if not handle then
    return pandoc.Para{pandoc.Str("Error: Could not execute _render_listing.py")}
  end

  local result = handle:read("*all")
  handle:close()

  -- Parse the markdown result and return as pandoc blocks
  if result and result ~= "" then
    local doc_result = pandoc.read(result, "markdown")
    return doc_result.blocks
  end

  -- Fallback if no result
  return pandoc.Para{pandoc.Str("No functions found matching pattern: " .. file_pattern)}
  end
}
