// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

use std::sync::Arc;

use datafusion_expr::{expr::ScalarFunction, Expr};
use savvy::{savvy, savvy_err};
use sedona::context::SedonaContext;

#[savvy]
pub struct SedonaDBExpr {
    pub inner: Expr,
}

#[savvy]
pub struct SedonaDBExprFactory {
    pub ctx: Arc<SedonaContext>,
}

#[savvy]
impl SedonaDBExprFactory {
    fn scalar_function(&self, name: &str, args: savvy::Sexp) -> savvy::Result<SedonaDBExpr> {
        if let Some(scalar_udf) = self.ctx.ctx.state().scalar_functions().get(name) {
            let args = Self::exprs(args)?;
            let inner = ScalarFunction::new_udf(scalar_udf.clone(), args);
            Ok(SedonaDBExpr {
                inner: Expr::ScalarFunction(inner),
            })
        } else {
            Err(savvy_err!("Scalar UDF '{name}' not found"))
        }
    }
}

impl SedonaDBExprFactory {
    fn exprs(exprs_sexp: savvy::Sexp) -> savvy::Result<Vec<Expr>> {
        savvy::ListSexp::try_from(exprs_sexp)?
            .iter()
            .map(|(_, item)| -> savvy::Result<Expr> {
                let expr_wrapper = SedonaDBExpr::try_from(item)?;
                Ok(expr_wrapper.inner.clone())
            })
            .collect()
    }
}
