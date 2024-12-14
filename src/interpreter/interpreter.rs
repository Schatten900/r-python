use std::collections::HashMap;

use crate::ir::ast::{Expression, Function, Name, Statement, Type};
use crate::tc::type_checker::{check_expression, check_statement};

type ErrorMessage = String;

pub enum Control {
    Continue(Environment),
    Return(Expression)
}

#[derive(Debug, Clone, PartialEq)]
pub enum EnvValue {
    Exp(Option<Expression>),
    Func(Function)
}

pub type Environment = HashMap<Name, (EnvValue, Type)>;

pub fn eval(exp: Expression, env: &Environment) -> Result<Expression, ErrorMessage> {
    check_expression(exp.clone(), env)?;

    match exp {
        Expression::Add(lhs, rhs) => add(*lhs, *rhs, env),
        Expression::Sub(lhs, rhs) => sub(*lhs, *rhs, env),
        Expression::Mul(lhs, rhs) => mul(*lhs, *rhs, env),
        Expression::Div(lhs, rhs) => div(*lhs, *rhs, env),
        Expression::Rmd(lhs, rhs) => rmd(*lhs, *rhs, env),
        Expression::And(lhs, rhs) => and(*lhs, *rhs, env),
        Expression::Or(lhs, rhs) => or(*lhs, *rhs, env),
        Expression::Not(lhs) => not(*lhs, env),
        Expression::EQ(lhs, rhs) => eq(*lhs, *rhs, env),
        Expression::GT(lhs, rhs) => gt(*lhs, *rhs, env),
        Expression::LT(lhs, rhs) => lt(*lhs, *rhs, env),
        Expression::GTE(lhs, rhs) => gte(*lhs, *rhs, env),
        Expression::LTE(lhs, rhs) => lte(*lhs, *rhs, env),
        Expression::Var(name) => lookup(name, env),
        Expression::FuncCall(name, args) => call(&name, args, env),
        _ if is_constant(exp.clone()) => Ok(exp),
        _ => Err(String::from("Not implemented yet.")),
    }
}

fn is_constant(exp: Expression) -> bool {
    match exp {
        Expression::CTrue => true,
        Expression::CFalse => true,
        Expression::CInt(_) => true,
        Expression::CReal(_) => true,
        Expression::CString(_) => true,
        _ => false,
    }
}

fn lookup(name: String, env: &Environment) -> Result<Expression, ErrorMessage> {
    match env.get(&name) {
        Some((EnvValue::Exp(Some(value)), _)) => {
            Ok(value.clone())
        }
        _ => Err(format!("Variable '{}' not found", name))
    }
}

/* Arithmetic Operations */
fn eval_binary_arith_op<F>(
    lhs: Expression,
    rhs: Expression,
    env: &Environment,
    op: F,
    error_msg: &str,
) -> Result<Expression, ErrorMessage>
where
    F: Fn(f64, f64) -> f64,
{
    let v1 = eval(lhs, env)?;
    let v2 = eval(rhs, env)?;
    match (v1, v2) {
        (Expression::CInt(v1), Expression::CInt(v2)) => {
            Ok(Expression::CInt(op(v1 as f64, v2 as f64) as i32))
        }
        (Expression::CInt(v1), Expression::CReal(v2)) => Ok(Expression::CReal(op(v1 as f64, v2))),
        (Expression::CReal(v1), Expression::CInt(v2)) => Ok(Expression::CReal(op(v1, v2 as f64))),
        (Expression::CReal(v1), Expression::CReal(v2)) => Ok(Expression::CReal(op(v1, v2))),
        _ => Err(error_msg.to_string()),
    }
}

fn add(lhs: Expression, rhs: Expression, env: &Environment) -> Result<Expression, ErrorMessage> {
    eval_binary_arith_op(
        lhs,
        rhs,
        env,
        |a, b| a + b,
        "addition '(+)' is only defined for numbers (integers and real).",
    )
}

fn sub(lhs: Expression, rhs: Expression, env: &Environment) -> Result<Expression, ErrorMessage> {
    eval_binary_arith_op(
        lhs,
        rhs,
        env,
        |a, b| a - b,
        "subtraction '(-)' is only defined for numbers (integers and real).",
    )
}

fn mul(lhs: Expression, rhs: Expression, env: &Environment) -> Result<Expression, ErrorMessage> {
    eval_binary_arith_op(
        lhs,
        rhs,
        env,
        |a, b| a * b,
        "multiplication '(*)' is only defined for numbers (integers and real).",
    )
}

fn div(lhs: Expression, rhs: Expression, env: &Environment) -> Result<Expression, ErrorMessage> {
    eval_binary_arith_op(
        lhs,
        rhs,
        env,
        |a, b| a / b,
        "division '(/)' is only defined for numbers (integers and real).",
    )
}

fn rmd(lhs: Expression, rhs: Expression, env: &Environment) -> Result<Expression, ErrorMessage> {
    eval_binary_arith_op(
        lhs,
        rhs,
        env,
        |a, b| a % b,
        "Remainder operation '(%)' is only defined for numbers (integers and real).",
    )
}
/* Boolean Expressions */
fn eval_binary_boolean_op<F>(
    lhs: Expression,
    rhs: Expression,
    env: &Environment,
    op: F,
    error_msg: &str,
) -> Result<Expression, ErrorMessage>
where
    F: Fn(bool, bool) -> Expression,
{
    let v1 = eval(lhs, env)?;
    let v2 = eval(rhs, env)?;
    match (v1, v2) {
        (Expression::CTrue, Expression::CTrue) => Ok(op(true, true)),
        (Expression::CTrue, Expression::CFalse) => Ok(op(true, false)),
        (Expression::CFalse, Expression::CTrue) => Ok(op(false, true)),
        (Expression::CFalse, Expression::CFalse) => Ok(op(false, false)),
        _ => Err(error_msg.to_string()),
    }
}

fn and(lhs: Expression, rhs: Expression, env: &Environment) -> Result<Expression, ErrorMessage> {
    eval_binary_boolean_op(
        lhs,
        rhs,
        env,
        |a, b| {
            if a && b {
                Expression::CTrue
            } else {
                Expression::CFalse
            }
        },
        "'and' is only defined for booleans.",
    )
}

fn or(lhs: Expression, rhs: Expression, env: &Environment) -> Result<Expression, ErrorMessage> {
    eval_binary_boolean_op(
        lhs,
        rhs,
        env,
        |a, b| {
            if a || b {
                Expression::CTrue
            } else {
                Expression::CFalse
            }
        },
        "'or' is only defined for booleans.",
    )
}

fn not(lhs: Expression, env: &Environment) -> Result<Expression, ErrorMessage> {
    let v = eval(lhs, env)?;
    match v {
        Expression::CTrue => Ok(Expression::CFalse),
        Expression::CFalse => Ok(Expression::CTrue),
        _ => Err(String::from("'not' is only defined for booleans.")),
    }
}

/* Relational Operations */
fn eval_binary_rel_op<F>(
    lhs: Expression,
    rhs: Expression,
    env: &Environment,
    op: F,
    error_msg: &str,
) -> Result<Expression, ErrorMessage>
where
    F: Fn(f64, f64) -> Expression,
{
    let v1 = eval(lhs, env)?;
    let v2 = eval(rhs, env)?;
    match (v1, v2) {
        (Expression::CInt(v1), Expression::CInt(v2)) => Ok(op(v1 as f64, v2 as f64)),
        (Expression::CInt(v1), Expression::CReal(v2)) => Ok(op(v1 as f64, v2)),
        (Expression::CReal(v1), Expression::CInt(v2)) => Ok(op(v1, v2 as f64)),
        (Expression::CReal(v1), Expression::CReal(v2)) => Ok(op(v1, v2)),
        _ => Err(error_msg.to_string()),
    }
}

fn eq(lhs: Expression, rhs: Expression, env: &Environment) -> Result<Expression, ErrorMessage> {
    eval_binary_rel_op(
        lhs,
        rhs,
        env,
        |a, b| {
            if a == b {
                Expression::CTrue
            } else {
                Expression::CFalse
            }
        },
        "(==) is only defined for numbers (integers and real).",
    )
}

fn gt(lhs: Expression, rhs: Expression, env: &Environment) -> Result<Expression, ErrorMessage> {
    eval_binary_rel_op(
        lhs,
        rhs,
        env,
        |a, b| {
            if a > b {
                Expression::CTrue
            } else {
                Expression::CFalse
            }
        },
        "(>) is only defined for numbers (integers and real).",
    )
}

fn lt(lhs: Expression, rhs: Expression, env: &Environment) -> Result<Expression, ErrorMessage> {
    eval_binary_rel_op(
        lhs,
        rhs,
        env,
        |a, b| {
            if a < b {
                Expression::CTrue
            } else {
                Expression::CFalse
            }
        },
        "(<) is only defined for numbers (integers and real).",
    )
}

fn gte(lhs: Expression, rhs: Expression, env: &Environment) -> Result<Expression, ErrorMessage> {
    eval_binary_rel_op(
        lhs,
        rhs,
        env,
        |a, b| {
            if a >= b {
                Expression::CTrue
            } else {
                Expression::CFalse
            }
        },
        "(>=) is only defined for numbers (integers and real).",
    )
}

fn lte(lhs: Expression, rhs: Expression, env: &Environment) -> Result<Expression, ErrorMessage> {
    eval_binary_rel_op(
        lhs,
        rhs,
        env,
        |a, b| {
            if a <= b {
                Expression::CTrue
            } else {
                Expression::CFalse
            }
        },
        "(<=) is only defined for numbers (integers and real).",
    )
}

/* Function calls */
fn call (
    name: &Name,
    args: Vec<Expression>,
    env: &Environment,
) -> Result<Expression, ErrorMessage> {
    if let Some((EnvValue::Func(func), _)) = env.get(name) {
        let mut local_env = env.clone();

        if let Some(params) = &func.params {
            if params.len() != args.len() {
                return Err(format!("'{}' expected {} arguments, got {}", name, params.len(), args.len()));
            }

            for (arg, (param, param_type)) in args.iter().zip(params) {
                let arg_value = eval(arg.clone(), &local_env)?;
                local_env.insert(param.to_string(), (EnvValue::Exp(Some(arg_value)), param_type.clone()));
            }
        }

        match execute(*func.body.clone(), local_env)? {
            Control::Return(value) => Ok(value),
            Control::Continue(_) => Err(format!("'{}' must have a return statement", name))
        }

    } else {
        Err(format!("'{}' is not defined", name))
    }
}

pub fn execute(stmt: Statement, env: Environment) -> Result<Control, ErrorMessage> {
    let mut new_env = env.clone();
    check_statement(stmt.clone(), &mut new_env)?;

    match stmt {
        Statement::Assignment(name, exp, kind) => {
            let value = eval(*exp, &new_env)?;
            new_env.insert(name.clone(), (EnvValue::Exp(Some(value)), kind));
            Ok(Control::Continue(new_env))
        }
        Statement::IfThenElse(cond, stmt_then, stmt_else) => {
            let value = eval(*cond, &new_env)?;
            match value {
                Expression::CTrue => execute(*stmt_then, new_env),
                Expression::CFalse => match stmt_else {
                    Some(else_statement) => execute(*else_statement, new_env),
                    None => Ok(Control::Continue(new_env)),
                },
                _ => Err(String::from("expecting a boolean value.")),
            }
        }
        Statement::While(cond, stmt) => {
            let mut value = eval(*cond.clone(), &new_env)?;

            while value == Expression::CTrue {
                let new_stmt = stmt.clone();
                match execute(*new_stmt, new_env)? {
                    Control::Continue(result_env) => {
                        value = eval(*cond.clone(), &result_env.clone())?;
                        new_env = result_env;
                    }
                    Control::Return(value) => return Ok(Control::Return(value))
                }
            }

            Ok(Control::Continue(new_env.clone()))
        }
        Statement::Sequence(s1, s2) => {
            match execute(*s1, new_env)? {
                Control::Continue(new_env) => {
                    execute(*s2, new_env)
                }
                Control::Return(value) => Ok(Control::Return(value))
            }
        }
        Statement::FuncDef(name, func, _) => {
            let mut stmt = *func.body.clone();
            let mut next_vec = vec![];
            loop {
                match stmt {
                    Statement::Return(_, _) => break,
                    Statement::Sequence(stmt1, stmt2) => {
                        stmt = *stmt1;
                        next_vec.push(*stmt2);
                    }
                    Statement::IfThenElse(_, stmt_then, option_stmt) => {
                        stmt = *stmt_then;
                        if let Some(stmt_else) = option_stmt {
                            next_vec.push(*stmt_else);
                        }
                    }
                    Statement::While(_, stmt_while) => stmt = *stmt_while,
                    _ => match next_vec.pop() {
                        Some(next_stmt) => stmt = next_stmt,
                        None => return Err(format!("'{}' must have a return statement", name))
                    }
                }
            }

            Ok(Control::Continue(new_env))
        }
        Statement::Return(_, exp) => {
            let value = eval(*exp, &new_env)?;
            Ok(Control::Return(value))
        }
        _ => Err(String::from("not implemented yet")),
    }
}

#[cfg(test)]
mod tests {

    use super::*;
    use crate::ir::ast::Expression::*;
    use crate::ir::ast::Statement::*;
    use crate::ir::ast::Type::*;
    use approx::relative_eq;

    #[test]
    fn eval_constant() {
        let env = HashMap::new();
        let c10 = CInt(10);
        let c20 = CInt(20);

        assert_eq!(eval(c10, &env), Ok(CInt(10)));
        assert_eq!(eval(c20, &env), Ok(CInt(20)));
    }

    #[test]
    fn eval_add_expression1() {
        let env = HashMap::new();
        let c10 = CInt(10);
        let c20 = CInt(20);
        let add1 = Add(Box::new(c10), Box::new(c20));
        assert_eq!(eval(add1, &env), Ok(CInt(30)));
    }

    #[test]
    fn eval_add_expression2() {
        let env = HashMap::new();
        let c10 = CInt(10);
        let c20 = CInt(20);
        let c30 = CInt(30);
        let add1 = Add(Box::new(c10), Box::new(c20));
        let add2 = Add(Box::new(add1), Box::new(c30));
        assert_eq!(eval(add2, &env), Ok(CInt(60)));
    }

    #[test]
    fn eval_add_expression3() {
        let env = HashMap::new();
        let c10 = CInt(10);
        let c20 = CReal(20.5);
        let add1 = Add(Box::new(c10), Box::new(c20));
        assert_eq!(eval(add1, &env), Ok(CReal(30.5)));
    }

    #[test]
    fn eval_sub_expression1() {
        let env = HashMap::new();
        let c10 = CInt(10);
        let c20 = CInt(20);
        let sub1 = Sub(Box::new(c20), Box::new(c10));
        assert_eq!(eval(sub1, &env), Ok(CInt(10)));
    }

    #[test]
    fn eval_sub_expression2() {
        let env = HashMap::new();
        let c100 = CInt(100);
        let c200 = CInt(300);
        let sub1 = Sub(Box::new(c200), Box::new(c100));
        assert_eq!(eval(sub1, &env), Ok(CInt(200)));
    }

    #[test]
    fn eval_sub_expression3() {
        let env = HashMap::new();
        let c100 = CReal(100.5);
        let c300 = CInt(300);
        let sub1 = Sub(Box::new(c300), Box::new(c100));
        assert_eq!(eval(sub1, &env), Ok(CReal(199.5)));
    }

    #[test]
    fn eval_mul_expression1() {
        let env = HashMap::new();
        let c10 = CInt(10);
        let c20 = CInt(20);
        let mul1 = Mul(Box::new(c10), Box::new(c20));
        assert_eq!(eval(mul1, &env), Ok(CInt(200)));
    }

    #[test]
    fn eval_mul_expression2() {
        let env = HashMap::new();
        let c10 = CReal(10.5);
        let c20 = CInt(20);
        let mul1 = Mul(Box::new(c10), Box::new(c20));
        assert_eq!(eval(mul1, &env), Ok(CReal(210.0)));
    }

    #[test]
    fn eval_div_expression1() {
        let env = HashMap::new();
        let c10 = CInt(10);
        let c20 = CInt(20);
        let div1 = Div(Box::new(c20), Box::new(c10));
        assert_eq!(eval(div1, &env), Ok(CInt(2)));
    }

    #[test]
    fn eval_div_expression2() {
        let env = HashMap::new();
        let c10 = CInt(10);
        let c3 = CInt(3);
        let div1 = Div(Box::new(c10), Box::new(c3));
        assert_eq!(eval(div1, &env), Ok(CInt(3)));
    }

    #[test]
    fn eval_div_expression3() {
        let env = HashMap::new();
        let c3 = CInt(3);
        let c21 = CInt(21);
        let div1 = Div(Box::new(c21), Box::new(c3));
        assert_eq!(eval(div1, &env), Ok(CInt(7)));
    }

    #[test]
    fn eval_div_expression4() {
        let env = HashMap::new();
        let c10 = CInt(10);
        let c3 = CReal(3.0);
        let div1 = Div(Box::new(c10), Box::new(c3));
        let res = eval(div1, &env);
        match res {
            Ok(CReal(v)) => assert!(relative_eq!(v, 3.3333333333333335, epsilon = f64::EPSILON)),
            Err(msg) => assert!(false, "{}", msg),
            _ => assert!(false, "Not expected."),
        }
    }

    #[test]
    fn eval_variable() {
        let env = HashMap::from([(String::from("x"), (EnvValue::Exp(Some(CInt(10))), TInteger)), (String::from("y"), (EnvValue::Exp(Some(CInt(20))), TInteger))]);
        let v1 = Var(String::from("x"));
        let v2 = Var(String::from("y"));
        assert_eq!(eval(v1, &env), Ok(CInt(10)));
        assert_eq!(eval(v2, &env), Ok(CInt(20)));
    }

    #[test]
    fn eval_expression_with_variables() {
        let env = HashMap::from([(String::from("a"), (EnvValue::Exp(Some(CInt(5))), TInteger)), (String::from("b"), (EnvValue::Exp(Some(CInt(3))), TInteger))]);
        let expr = Mul(
            Box::new(Var(String::from("a"))),
            Box::new(Add(Box::new(Var(String::from("b"))), Box::new(CInt(2)))),
        );
        assert_eq!(eval(expr, &env), Ok(CInt(25)));
    }

    #[test]
    fn eval_nested_expressions() {
        let env = HashMap::new();
        let expr = Add(
            Box::new(Mul(Box::new(CInt(2)), Box::new(CInt(3)))),
            Box::new(Sub(Box::new(CInt(10)), Box::new(CInt(4)))),
        );
        assert_eq!(eval(expr, &env), Ok(CInt(12)));
    }

    #[test]
    fn eval_variable_not_found() {
        let env = HashMap::new();
        let var_expr = Var(String::from("z"));

        assert_eq!(
            eval(var_expr, &env),
            Err(String::from("Variable 'z' not found"))
        );
    }

    #[test]
    fn execute_assignment() {
        let env = HashMap::new();
        let assign_stmt = Assignment(String::from("x"), Box::new(CInt(42)), TInteger);

        match execute(assign_stmt, env) {
            Ok(Control::Continue(new_env)) => assert_eq!(new_env.get("x"), Some(&(EnvValue::Exp(Some(CInt(42))), TInteger))),
            Ok(Control::Return(_)) => assert!(false),
            Err(s) => assert!(false, "{}", s),
        }
    }

    #[test]
    fn eval_summation() {
        /*
         * (a test case for the following program)
         *
         * > x = 10
         * > y = 0
         * > while x >= 0:
         * >   y = y + x
         * >   x = x - 1
         *
         * After executing this program, 'x' must be zero and
         * 'y' must be 55.
         */
        let env = HashMap::new();

        let a1 = Assignment(String::from("x"), Box::new(CInt(10)), TInteger);
        let a2 = Assignment(String::from("y"), Box::new(CInt(0)), TInteger);
        let a3 = Assignment(
            String::from("y"),
            Box::new(Add(
                Box::new(Var(String::from("y"))),
                Box::new(Var(String::from("x"))),
            )),
            TInteger
        );
        let a4 = Assignment(
            String::from("x"),
            Box::new(Sub(Box::new(Var(String::from("x"))), Box::new(CInt(1)))),
            TInteger
        );

        let seq1 = Sequence(Box::new(a3), Box::new(a4));

        let while_statement = While(
            Box::new(GT(Box::new(Var(String::from("x"))), Box::new(CInt(0)))),
            Box::new(seq1),
        );

        let seq2 = Sequence(Box::new(a2), Box::new(while_statement));
        let program = Sequence(Box::new(a1), Box::new(seq2));

        match execute(program, env) {
            Ok(Control::Continue(new_env)) => {
                assert_eq!(new_env.get("y"), Some(&(EnvValue::Exp(Some(CInt(55))), TInteger)));
                assert_eq!(new_env.get("x"), Some(&(EnvValue::Exp(Some(CInt(0))), TInteger)));
            }
            Ok(Control::Return(_)) => assert!(false),
            Err(s) => assert!(false, "{}", s),
        }
    }

    #[test]
    fn eval_simple_if_then_else() {
        /*
         * Test for simple if-then-else statement
         *
         * > x = 10
         * > if x > 5:
         * >   y = 1
         * > else:
         * >   y = 0
         *
         * After executing, 'y' should be 1.
         */
        let env = HashMap::new();

        let condition = GT(Box::new(Var(String::from("x"))), Box::new(CInt(5)));
        let then_stmt = Assignment(String::from("y"), Box::new(CInt(1)), TInteger);
        let else_stmt = Assignment(String::from("y"), Box::new(CInt(0)), TInteger);

        let if_statement = IfThenElse(
            Box::new(condition),
            Box::new(then_stmt),
            Some(Box::new(else_stmt)),
        );

        let setup_stmt = Assignment(String::from("x"), Box::new(CInt(10)), TInteger);
        let program = Sequence(Box::new(setup_stmt), Box::new(if_statement));

        match execute(program, env) {
            Ok(Control::Continue(new_env)) => assert_eq!(new_env.get("y"), Some(&(EnvValue::Exp(Some(CInt(1))), TInteger))),
            Ok(Control::Return(_)) => assert!(false),
            Err(s) => assert!(false, "{}", s),
        }
    }

    #[test]
    fn eval_if_then_optional_else() {
        /*
         * Test for simple if-then-else statement
         *
         * > x = 1
         * > y = 0
         * > if x == y:
         * >   y = 1
         * > else:
         * >    y = 2
         * >    if x < 0:
         * >        y = 5
         *
         * After executing, 'y' should be 2.
         */

        let env = HashMap::new();

        let second_condition = LT(Box::new(Var(String::from("x"))), Box::new(CInt(0)));
        let second_then_stmt = Assignment(String::from("y"), Box::new(CInt(5)), TInteger);

        let second_if_stmt =
            IfThenElse(Box::new(second_condition), Box::new(second_then_stmt), None);

        let else_setup_stmt = Assignment(String::from("y"), Box::new(CInt(2)), TInteger);
        let else_stmt = Sequence(Box::new(else_setup_stmt), Box::new(second_if_stmt));

        let first_condition = EQ(
            Box::new(Var(String::from("x"))),
            Box::new(Var(String::from("y"))),
        );
        let first_then_stmt = Assignment(String::from("y"), Box::new(CInt(1)), TInteger);

        let first_if_stmt = IfThenElse(
            Box::new(first_condition),
            Box::new(first_then_stmt),
            Some(Box::new(else_stmt)),
        );

        let second_assignment = Assignment(String::from("y"), Box::new(CInt(0)), TInteger);
        let setup_stmt = Sequence(Box::new(second_assignment), Box::new(first_if_stmt));

        let first_assignment = Assignment(String::from("x"), Box::new(CInt(1)), TInteger);
        let program = Sequence(Box::new(first_assignment), Box::new(setup_stmt));

        match execute(program, env) {
            Ok(Control::Continue(new_env)) => assert_eq!(new_env.get("y"), Some(&(EnvValue::Exp(Some(CInt(2))), TInteger))),
            Ok(Control::Return(_)) => assert!(false),
            Err(s) => assert!(false, "{}", s),
        }
    }

    #[test]
    fn eval_while_using_rmd() {
        /*
         *   Test for remainder operator using while
         *
         *   x = 1
         *   y = 1800
         *   z = 0
         *   while x*x <= y:
         *       if y % x == 0:
         *           if x % 2 == 0:
         *               z = z + 1
         *           if (y / x) % 2 == 0:
         *               z = z + 1
         *       x = x + 1
         *
         *   After processing 'x' must be 43
         *   and 'z' must be 27
         */

        let env = HashMap::new();

        let a1 = Assignment(String::from("x"), Box::new(CInt(1)), TInteger);
        let a2 = Assignment(String::from("y"), Box::new(CInt(1800)), TInteger);
        let a3 = Assignment(String::from("z"), Box::new(CInt(0)), TInteger);
        let a4 = Assignment(
            String::from("z"),
            Box::new(Add(Box::new(Var(String::from("z"))), Box::new(CInt(1)))),
            TInteger
        );
        let a5 = Assignment(
            String::from("z"),
            Box::new(Add(Box::new(Var(String::from("z"))), Box::new(CInt(1)))),
            TInteger
        );
        let a6 = Assignment(
            String::from("x"),
            Box::new(Add(Box::new(Var(String::from("x"))), Box::new(CInt(1)))),
            TInteger
        );

        let if_statement1 = IfThenElse(
            Box::new(EQ(
                Box::new(Rmd(Box::new(Var(String::from("x"))), Box::new(CInt(2)))),
                Box::new(CInt(0)),
            )),
            Box::new(a4),
            None,
        );
        let if_statement2 = IfThenElse(
            Box::new(EQ(
                Box::new(Rmd(
                    Box::new(Div(
                        Box::new(Var(String::from("y"))),
                        Box::new(Var(String::from("x"))),
                    )),
                    Box::new(CInt(2)),
                )),
                Box::new(CInt(0)),
            )),
            Box::new(a5),
            None,
        );

        let seq = Sequence(Box::new(if_statement1), Box::new(if_statement2));
        let if_statement = IfThenElse(
            Box::new(EQ(
                Box::new(Rmd(
                    Box::new(Var(String::from("y"))),
                    Box::new(Var(String::from("x"))),
                )),
                Box::new(CInt(0)),
            )),
            Box::new(seq),
            None,
        );

        let seq1 = Sequence(Box::new(if_statement), Box::new(a6));

        let while_statement = While(
            Box::new(LTE(
                Box::new(Mul(
                    Box::new(Var(String::from("x"))),
                    Box::new(Var(String::from("x"))),
                )),
                Box::new(Var(String::from("y"))),
            )),
            Box::new(seq1),
        );

        let seq2 = Sequence(Box::new(Sequence(Box::new(a1), Box::new(a2))), Box::new(a3));

        let program = Sequence(Box::new(seq2), Box::new(while_statement));

        match execute(program, env) {
            Ok(Control::Continue(new_env)) => {
                assert_eq!(new_env.get("x"), Some(&(EnvValue::Exp(Some(CInt(43))), TInteger)));
                assert_eq!(new_env.get("z"), Some(&(EnvValue::Exp(Some(CInt(27))), TInteger)));
            }
            Ok(Control::Return(_)) => assert!(false),
            Err(s) => assert!(false, "{}", s),
        }
    }

    #[test]
    fn eval_while_with_if() {
        /*
         *   Test for more complex while statement
         *
         *   x = 1
         *   y = 16
         *   z = 16
         *   a = 0
         *   while x <= y && a*a != z:
         *       m = (x + y) / 2
         *       if m*m <= z:
         *          a = m
         *          x = mid + 1
         *       else:
         *          y = mid - 1
         *
         *   After executing this program, 'x' must be 5,
         *   'y' must be 7 and 'a' must be 4
         */

        let env = HashMap::new();

        let a1 = Assignment(String::from("x"), Box::new(CInt(1)), TInteger);
        let a2 = Assignment(String::from("y"), Box::new(CInt(16)), TInteger);
        let a3 = Assignment(String::from("z"), Box::new(CInt(16)), TInteger);
        let a4 = Assignment(String::from("a"), Box::new(CInt(0)), TInteger);
        let a5 = Assignment(
            String::from("m"),
            Box::new(Div(
                Box::new(Add(
                    Box::new(Var(String::from("x"))),
                    Box::new(Var(String::from("y"))),
                )),
                Box::new(CInt(2)),
            )),
            TInteger
        );
        let a6 = Assignment(String::from("a"), Box::new(Var(String::from("m"))), TInteger);
        let a7 = Assignment(
            String::from("x"),
            Box::new(Add(Box::new(Var(String::from("m"))), Box::new(CInt(1)))),
            TInteger
        );
        let a8 = Assignment(
            String::from("y"),
            Box::new(Sub(Box::new(Var(String::from("m"))), Box::new(CInt(1)))),
            TInteger
        );

        let seq = Sequence(Box::new(a6), Box::new(a7));

        let if_statement: Statement = IfThenElse(
            Box::new(LTE(
                Box::new(Mul(
                    Box::new(Var(String::from("m"))),
                    Box::new(Var(String::from("m"))),
                )),
                Box::new(Var(String::from("z"))),
            )),
            Box::new(seq),
            Some(Box::new(a8)),
        );

        let while_statement = While(
            Box::new(And(
                Box::new(LTE(
                    Box::new(Var(String::from("x"))),
                    Box::new(Var(String::from("y"))),
                )),
                Box::new(Not(Box::new(EQ(
                    Box::new(Mul(
                        Box::new(Var(String::from("a"))),
                        Box::new(Var(String::from("a"))),
                    )),
                    Box::new(Var(String::from("z"))),
                )))),
            )),
            Box::new(Sequence(Box::new(a5), Box::new(if_statement))),
        );

        let seq1 = Sequence(
            Box::new(a1),
            Box::new(Sequence(
                Box::new(a2),
                Box::new(Sequence(Box::new(a3), Box::new(a4))),
            )),
        );

        let program = Sequence(Box::new(seq1), Box::new(while_statement));

        match execute(program, env) {
            Ok(Control::Continue(new_env)) => {
                assert_eq!(new_env.get("x"), Some(&(EnvValue::Exp(Some(CInt(5))), TInteger)));
                assert_eq!(new_env.get("y"), Some(&(EnvValue::Exp(Some(CInt(7))), TInteger)));
                assert_eq!(new_env.get("a"), Some(&(EnvValue::Exp(Some(CInt(4))), TInteger)));
            }
            Ok(Control::Return(_)) => assert!(false),
            Err(s) => assert!(false, "{}", s),
        }
    }

    #[test]
    fn eval_while_with_boolean() {
        /*  Test for while statement using booleans
         *
         *   x = true
         *   y = 1
         *   while x:
         *       if y > 1e9:
         *           x = false
         *       else:
         *           y = y * 2
         *
         *   After executing this program 'y' must be equal 1073741824
         *   and 'x' must be equal false
         */

        let env = HashMap::new();

        let a1 = Assignment(String::from("x"), Box::new(CTrue), TBool);
        let a2 = Assignment(String::from("y"), Box::new(CInt(1)), TInteger);
        let a3 = Assignment(String::from("x"), Box::new(CFalse), TBool);
        let a4 = Assignment(
            String::from("y"),
            Box::new(Mul(Box::new(Var(String::from("y"))), Box::new(CInt(2)))),
            TInteger
        );

        let if_then_else_statement = IfThenElse(
            Box::new(GT(
                Box::new(Var(String::from("y"))),
                Box::new(CInt(1000000000)),
            )),
            Box::new(a3),
            Some(Box::new(a4)),
        );

        let while_statement = While(
            Box::new(Var(String::from("x"))),
            Box::new(if_then_else_statement),
        );

        let program = Sequence(
            Box::new(a1),
            Box::new(Sequence(Box::new(a2), Box::new(while_statement))),
        );

        match execute(program, env) {
            Ok(Control::Continue(new_env)) => {
                assert_eq!(new_env.get("x"), Some(&(EnvValue::Exp(Some(CFalse)), TBool)));
                assert_eq!(new_env.get("y"), Some(&(EnvValue::Exp(Some(CInt(1073741824))), TInteger)));
            }
            Ok(Control::Return(_)) => assert!(false),
            Err(s) => assert!(false, "{}", s),
        }
    }

    // #[test]
    // fn eval_while_loop_decrement() {
    //     /*
    //      * Test for while loop that decrements a variable
    //      *
    //      * > x = 3
    //      * > y = 10
    //      * > while x:
    //      * >   y = y - 1
    //      * >   x = x - 1
    //      *
    //      * After executing, 'y' should be 7 and 'x' should be 0.
    //      */
    //     let env = HashMap::new();
    //     let func_env = HashMap::new();

    //     let a1 = Statement::Assignment(Box::new(String::from("x")), Box::new(CInt(3)));
    //     let a2 = Statement::Assignment(Box::new(String::from("y")), Box::new(CInt(10)));
    //     let a3 = Statement::Assignment(
    //         Box::new(String::from("y")),
    //         Box::new(Sub(Box::new(Var(String::from("y"))), Box::new(CInt(1)))),
    //     );
    //     let a4 = Statement::Assignment(
    //         Box::new(String::from("x")),
    //         Box::new(Sub(
    //             Box::new(Var(String::from("x"))),
    //             Box::new(CInt(1)),
    //         )),
    //     );

    //     let seq1 = Statement::Sequence(Box::new(a3), Box::new(a4));
    //     let while_statement =
    //         Statement::While(Box::new(Var(String::from("x"))), Box::new(seq1));
    //     let program = Statement::Sequence(
    //         Box::new(a1),
    //         Box::new(Sequence(Box::new(a2), Box::new(while_statement))),
    //     );

    //     match execute(program, env, func_env) {
    //         Ok((new_env, new_func_env)) => {
    //             assert_eq!(new_env.get("y"), Some(&CInt(7)));
    //             assert_eq!(new_env.get("x"), Some(&CInt(0)));
    //         }
    //         Err(s) => assert!(false, "{}", s),
    //     }
    // }
    // #[test]
    // fn eval_nested_if_statements() {
    //     /*
    //      * Test for nested if-then-else statements
    //      *
    //      * > x = 10
    //      * > if x > 5:
    //      * >   if x > 8:
    //      * >     y = 1
    //      * >   else:
    //      * >     y = 2
    //      * > else:
    //      * >   y = 0
    //      *
    //      * After executing, 'y' should be 1.
    //      */
    //     let env = HashMap::new();
    //     let func_env = HashMap::new();

    //     let inner_then_stmt =
    //         Assignment(String::from("y")), Box:new(CInt(1)));
    //     let inner_else_stmt =
    //         Assignment(String::from("y")), Box:new(CInt(2)));
    //     let inner_if_statement = Statement::IfThenElse(
    //         Box::new(Var(String::from("x"))),
    //         Box::new(inner_then_stmt),
    //         Box::new(inner_else_stmt),
    //     );

    //     let outer_else_stmt =
    //         Assignment(String::from("y")), Box:new(CInt(0)));
    //     let outer_if_statement = Statement::IfThenElse(
    //         Box::new(Var(String::from("x"))),
    //         Box::new(inner_if_statement),
    //         Box::new(outer_else_stmt),
    //     );

    //     let setup_stmt =
    //         Assignment(String::from("x")), Box:new(CInt(10)));
    //     let program = Sequence(Box::new(setup_stmt), Box::new(outer_if_statement));

    //     match execute(&program, env, func_env) {
    //         Ok((new_env, new_func_env)) => assert_eq!(new_env.get("y"), Some(&1)),
    //         Err(s) => assert!(false, "{}", s),
    //     }
    // }

    #[test]
    fn eval_complex_sequence() {
        /*
         * Sequence with multiple assignments and expressions
         *
         * > x = 5
         * > y = 0
         * > z = 2 * x + 3
         *
         * After executing, 'x' should be 5, 'y' should be 0, and 'z' should be 13.
         */
        let env = HashMap::new();

        let a1 = Assignment(String::from("x"), Box::new(CInt(5)), TInteger);
        let a2 = Assignment(String::from("y"), Box::new(CInt(0)), TInteger);
        let a3 = Assignment(
            String::from("z"),
            Box::new(Add(
                Box::new(Mul(Box::new(CInt(2)), Box::new(Var(String::from("x"))))),
                Box::new(CInt(3)),
            )),
            TInteger
        );

        let program = Sequence(Box::new(a1), Box::new(Sequence(Box::new(a2), Box::new(a3))));

        match execute(program, env) {
            Ok(Control::Continue(new_env)) => {
                assert_eq!(new_env.get("x"), Some(&(EnvValue::Exp(Some(CInt(5))), TInteger)));
                assert_eq!(new_env.get("y"), Some(&(EnvValue::Exp(Some(CInt(0))), TInteger)));
                assert_eq!(new_env.get("z"), Some(&(EnvValue::Exp(Some(CInt(13))), TInteger)));
            }
            Ok(Control::Return(_)) => assert!(false),
            Err(s) => assert!(false, "{}", s),
        }
    }

    #[test]
    fn func_decl_call() {
        /*
         * Test for declaration and call of a function
         *
         * > def add(a: TInteger, b: TInteger) -> TInteger:
         * >    t = a + b
         * >    return t
         * >
         * > sum = add(5, 7)
         *
         * After executing, 'sum' should be 12.
         */
        let env = Environment::new();

        let program = Sequence(
            Box::new(FuncDef(
                String::from("add"), 
                Function {params: Some(vec![(String::from("a"), TInteger), (String::from("b"), TInteger)]), 
                          body: Box::new(Sequence(Box::new(Assignment(String::from("t"), Box::new(Add(Box::new(Var(String::from("a"))), Box::new(Var(String::from("b"))))), TInteger)), 
                                                  Box::new(Return(String::from("add"), Box::new(Var(String::from("t")))))))
                         },
                TInteger),
            ),
            Box::new(Assignment(String::from("sum"), Box::new(FuncCall(String::from("add"), vec![CInt(5), CInt(7)])), TInteger))
        );

        match execute(program, env) {
            Ok(Control::Continue(new_env)) => match new_env.get("sum") {
                Some((EnvValue::Exp(Some(CInt(12))), TInteger)) => {}
                Some(val) => assert!(false, "Expected 12, got {:?}", val),
                None => assert!(false, "Variable sum not found"),
            }
            Ok(Control::Return(_)) => assert!(false),
            Err(s) => assert!(false, "{}", s),
        }
    }

    #[test]
    fn recursive_func() {
        /*
         * Test for declaration and call of a recursive function
         *
         * > def fibonacci(n: TInteger) -> TInteger:
         * >    if n <= 2:
         * >        return n - 1
         * >    else:
         * >        return fib(n-1) + fib(n-2)
         * >
         * > fib = fibonacci(10)
         *
         * After executing, 'fib' should be 34.
         */
        let env = Environment::new();

        let program = Sequence(
            Box::new(FuncDef(
                String::from("fibonacci"), 
                Function {params: Some(vec![(String::from("n"), TInteger)]), 
                          body: Box::new(IfThenElse(Box::new(LTE(Box::new(Var(String::from("n"))), Box::new(CInt(2)))), 
                          Box::new(Return(String::from("fibonacci"), Box::new(Sub(Box::new(Var(String::from("n"))), Box::new(CInt(1)))))), 
                          Some(Box::new(Return(String::from("fibonacci"), Box::new(Add(Box::new(FuncCall(String::from("fibonacci"), vec![Sub(Box::new(Var(String::from("n"))), Box::new(CInt(1)))])), Box::new(FuncCall(String::from("fibonacci"), vec![Sub(Box::new(Var(String::from("n"))), Box::new(CInt(2)))])))))))))
                         },
                TInteger)
            ),
            Box::new(Assignment(String::from("fib"), Box::new(FuncCall(String::from("fibonacci"), vec![CInt(10)])), TInteger))
        );

        match execute(program, env) {
            Ok(Control::Continue(new_env)) => match new_env.get("fib") {
                Some((EnvValue::Exp(Some(CInt(34))), TInteger)) => {}
                Some(val) => assert!(false, "Expected 34, got {:?}", val),
                None => assert!(false, "Variable fib not found"),
            }
            Ok(Control::Return(_)) => assert!(false),
            Err(s) => assert!(false, "{}", s),
        }
    }

    #[test]
    fn func_call_args_error1() {
        /*
         * Test for call of a function with not enough arguments
         *
         * > def add(a: TInteger, b: TInteger) -> TInteger:
         * >    t = a + b
         * >    return t
         * >
         * > sum = add(5)
         */
        let env = Environment::new();

        let program = Sequence(
            Box::new(FuncDef(
                String::from("add"), 
                Function {params: Some(vec![(String::from("a"), TInteger), (String::from("b"), TInteger)]), 
                          body: Box::new(Sequence(Box::new(Assignment(String::from("t"), Box::new(Add(Box::new(Var(String::from("a"))), Box::new(Var(String::from("b"))))), TInteger)), 
                                                  Box::new(Return(String::from("add"), Box::new(Var(String::from("t")))))))
                         },
                TInteger)
            ),
            Box::new(Assignment(String::from("sum"), Box::new(FuncCall(String::from("add"), vec![CInt(5)])), TInteger))
        );

        match execute(program, env) {
            Ok(Control::Continue(_)) => assert!(false),
            Ok(Control::Return(_)) => assert!(false),
            Err(s) => assert_eq!(s, String::from("'add' expected 2 arguments, got 1")),
        }
    }

    #[test]
    fn func_call_args_error2() {
        /*
         * Test for call of a function with too many arguments
         *
         * > def add(a: TInteger, b: TInteger) -> TInteger:
         * >    t = a + b
         * >    return t
         * >
         * > sum = add(5, 7, 10)
         */
        let env = Environment::new();

        let program = Sequence(
            Box::new(FuncDef(
                String::from("add"), 
                Function {params: Some(vec![(String::from("a"), TInteger), (String::from("b"), TInteger)]), 
                          body: Box::new(Sequence(Box::new(Assignment(String::from("t"), Box::new(Add(Box::new(Var(String::from("a"))), Box::new(Var(String::from("b"))))), TInteger)), 
                                                  Box::new(Return(String::from("add"), Box::new(Var(String::from("t")))))))
                         },
                TInteger)
            ),
            Box::new(Assignment(String::from("sum"), Box::new(FuncCall(String::from("add"), vec![CInt(5), CInt(7), CInt(10)])), TInteger))
        );

        match execute(program, env) {
            Ok(Control::Continue(_)) => assert!(false),
            Ok(Control::Return(_)) => assert!(false),
            Err(s) => assert_eq!(s, String::from("'add' expected 2 arguments, got 3")),
        }
    }

    #[test]
    fn func_call_no_return() {
        /*
         * Test for declaration of a function with no return
         *
         * > def add(a: TInteger, b: TInteger) -> TInteger:
         * >    t = a + b
         * >
         * > sum = add(5, 7)
         */
        let env = Environment::new();

        let program = Sequence(
            Box::new(FuncDef(
                String::from("add"), 
                Function {params: Some(vec![(String::from("a"), TInteger), (String::from("b"), TInteger)]), 
                          body: Box::new(Assignment(String::from("t"), Box::new(Add(Box::new(Var(String::from("a"))), Box::new(Var(String::from("b"))))), TInteger))
                         },
                TInteger)
            ),
            Box::new(Assignment(String::from("sum"), Box::new(FuncCall(String::from("add"), vec![CInt(5), CInt(7)])), TInteger))
        );

        match execute(program, env) {
            Ok(Control::Continue(_)) => assert!(false),
            Ok(Control::Return(_)) => assert!(false),
            Err(s) => assert_eq!(s, String::from("'add' must have a return statement")),
        }
    }

    #[test]
    fn undefined_func_call() {
        /*
         * Test for call of an undefined function
         *
         * > sum = add(5, 7)
         */
        let env = Environment::new();

        let program = Assignment(String::from("sum"), Box::new(FuncCall(String::from("add"), vec![CInt(5), CInt(7)])), TInteger);

        match execute(program, env) {
            Ok(Control::Continue(_)) => assert!(false),
            Ok(Control::Return(_)) => assert!(false),
            Err(s) => assert_eq!(s, String::from("'add' is not defined")),
        }
    }
}