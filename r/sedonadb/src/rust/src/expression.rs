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

use datafusion_common::{Column, Result, ScalarValue};
use datafusion_expr::{
    expr::{AggregateFunction, FieldMetadata, NullTreatment, ScalarFunction},
    BinaryExpr, Cast, Expr, Operator,
};
use savvy::{savvy, savvy_err, EnvironmentSexp};
use sedona::context::SedonaContext;

use crate::{
    context::InternalContext,
    ffi::{import_array, import_field},
};

#[savvy]
pub struct SedonaDBExpr {
    pub inner: Expr,
}

#[savvy]
impl SedonaDBExpr {
    fn display(&self) -> savvy::Result<savvy::Sexp> {
        format!("{}", self.inner).try_into()
    }

    fn debug_string(&self) -> savvy::Result<savvy::Sexp> {
        format!("{:?}", self.inner).try_into()
    }

    fn alias(&self, name: &str) -> savvy::Result<SedonaDBExpr> {
        let inner = self.inner.clone().alias_if_changed(name.to_string())?;
        Ok(Self { inner })
    }

    fn cast(&self, schema_xptr: savvy::Sexp) -> savvy::Result<SedonaDBExpr> {
        let field = import_field(schema_xptr)?;
        if let Some(type_name) = field.extension_type_name() {
            return Err(savvy_err!(
                "Can't cast to Arrow extension type '{type_name}'"
            ));
        }

        let inner = Expr::Cast(Cast::new(
            self.inner.clone().into(),
            field.data_type().clone(),
        ));

        Ok(Self { inner })
    }

    fn negate(&self) -> savvy::Result<SedonaDBExpr> {
        let inner = Expr::Negative(Box::new(self.inner.clone()));
        Ok(Self { inner })
    }
}

#[savvy]
pub struct SedonaDBExprFactory {
    pub ctx: Arc<SedonaContext>,
}

#[savvy]
impl SedonaDBExprFactory {
    fn new(ctx: &InternalContext) -> Self {
        Self {
            ctx: ctx.inner.clone(),
        }
    }

    fn literal(array_xptr: savvy::Sexp, schema_xptr: savvy::Sexp) -> savvy::Result<SedonaDBExpr> {
        let (field, array_ref) = import_array(array_xptr, schema_xptr)?;
        let metadata = if field.metadata().is_empty() {
            None
        } else {
            Some(FieldMetadata::new_from_field(&field))
        };

        let scalar_value = ScalarValue::try_from_array(&array_ref, 0)?;
        let inner = Expr::Literal(scalar_value, metadata);
        Ok(SedonaDBExpr { inner })
    }

    fn column(&self, name: &str, qualifier: Option<&str>) -> savvy::Result<SedonaDBExpr> {
        let inner = Expr::Column(Column::new(qualifier, name));
        Ok(SedonaDBExpr { inner })
    }

    fn binary(
        &self,
        op: &str,
        lhs: &SedonaDBExpr,
        rhs: &SedonaDBExpr,
    ) -> savvy::Result<SedonaDBExpr> {
        let operator = match op {
            "==" => Operator::Eq,
            "!=" => Operator::NotEq,
            ">" => Operator::Gt,
            ">=" => Operator::GtEq,
            "<" => Operator::Lt,
            "<=" => Operator::LtEq,
            "+" => Operator::Plus,
            "-" => Operator::Minus,
            "*" => Operator::Multiply,
            "/" => Operator::Divide,
            "&" => Operator::And,
            "|" => Operator::Or,
            other => return Err(savvy_err!("Unimplemented binary operation '{other}'")),
        };

        let inner = Expr::BinaryExpr(BinaryExpr::new(
            Box::new(lhs.inner.clone()),
            operator,
            Box::new(rhs.inner.clone()),
        ));
        Ok(SedonaDBExpr { inner })
    }

    fn scalar_function(&self, name: &str, args: savvy::Sexp) -> savvy::Result<SedonaDBExpr> {
        if let Some(udf) = self.ctx.ctx.state().scalar_functions().get(name) {
            let args = Self::exprs(args)?;
            let inner = Expr::ScalarFunction(ScalarFunction::new_udf(udf.clone(), args));
            Ok(SedonaDBExpr { inner })
        } else {
            Err(savvy_err!("Scalar UDF '{name}' not found"))
        }
    }

    fn aggregate_function(
        &self,
        name: &str,
        args: savvy::Sexp,
        na_rm: Option<bool>,
        distinct: Option<bool>,
    ) -> savvy::Result<SedonaDBExpr> {
        if let Some(udf) = self.ctx.ctx.state().aggregate_functions().get(name) {
            let args = Self::exprs(args)?;
            let null_treatment = if na_rm.unwrap_or(true) {
                NullTreatment::IgnoreNulls
            } else {
                NullTreatment::RespectNulls
            };

            let inner = Expr::AggregateFunction(AggregateFunction::new_udf(
                udf.clone(),
                args,
                distinct.unwrap_or(false),
                None,   // filter
                vec![], // order by
                Some(null_treatment),
            ));

            Ok(SedonaDBExpr { inner })
        } else {
            Err(savvy_err!("Aggregate UDF '{name}' not found"))
        }
    }
}

impl SedonaDBExprFactory {
    pub fn exprs(exprs_sexp: savvy::Sexp) -> savvy::Result<Vec<Expr>> {
        savvy::ListSexp::try_from(exprs_sexp)?
            .iter()
            .map(|(_, item)| -> savvy::Result<Expr> {
                // item here is the Environment wrapper around the external pointer
                let expr_wrapper: &SedonaDBExpr = EnvironmentSexp::try_from(item)?.try_into()?;
                Ok(expr_wrapper.inner.clone())
            })
            .collect()
    }
}

impl TryFrom<EnvironmentSexp> for &SedonaDBExpr {
    type Error = savvy::Error;

    fn try_from(env: EnvironmentSexp) -> Result<Self, Self::Error> {
        env.get(".ptr")?
            .map(<&SedonaDBExpr>::try_from)
            .transpose()?
            .ok_or(savvy_err!("Invalid SedonaDBExpr object."))
    }
}
