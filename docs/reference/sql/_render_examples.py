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

import sedonadb

sd = sedonadb.connect()


def render_examples(examples, width=80, ascii=False):
    """Renders examples to stdout using SedonaDB for Python

    This takes an iterable of example SQL strings and renders the SQL
    and the result to stdout. All examples are run in the same
    context.

    Output:

    ````

    ```sql
    --- examples that don't return a result are accumulated
    SET some.config = true;
    --- when a query does return a result, we end the sql code block
    --- and print the result
    SELECT one as 1;
    ```

    ```
    <output>
    ```

    ````
    """
    try:
        examples_iter = iter(examples)
        while True:
            render_examples_iter_until_result(examples_iter, width=width, ascii=ascii)
    except StopIteration:
        pass


def render_examples_iter_until_result(examples_iter, width=80, ascii=False):
    example = next(examples_iter)

    # Open the block where the SQL is printed
    print("\n```sql")
    while example is not None:
        # Execute the example to get a row count. If this is a resultless
        # statement (no rows, no cols), don't execute it again to print. This
        # allows SET and CREATE TABLE statements to "set up" examples. We
        # could also look for a trailing semicolon (e.g., only print results
        # of statements without a trailing semicolon) if this approach is
        # problematic.

        # Echo the example
        print(example.strip())

        # Parse it
        df = sd.sql(example)

        # Execute and check emptiness
        if df.execute() == 0 and not df.schema.names:
            example = next(examples_iter, None)
            continue

        # Close the ```sql block
        print("```\n")

        # Print the result block (executes the query again)
        print("```")
        df.show(limit=None, width=width, ascii=ascii)
        print("```")
        return

    # If we're here, none of the statements had any output, so we need to close
    # the sql block
    print("```\n")


if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Render SedonaDB SQL examples")
    parser.add_argument(
        "examples",
        nargs="+",
        help=(
            "SQL strings to be rendered or `-` to read from stdin. "
            "When reading from stdin, multiple examples may be separated by "
            "with `----` on its own line."
        ),
    )
    parser.add_argument("--width", type=int, default=80)
    parser.add_argument("--ascii", default=False, action="store_true")

    args = parser.parse_args(sys.argv[1:])
    if args.examples == ["-"]:
        args.examples = sys.stdin.read().split("\n----\n")

    try:
        render_examples(args.examples, width=args.width, ascii=args.ascii)
    except Exception as e:
        raise ValueError(
            f"Failed to render examples:\n{'\n----\n'.join(args.examples)}"
        ) from e
