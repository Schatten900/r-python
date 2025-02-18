use crate::ir::ast::{EnvValue, Environment, Expression, Name, Statement, Type};
use std::collections::HashMap;
type ErrorMessage = String;


pub enum ControlType {
    Continue(Environment),
    Return(Type),
}

pub fn check_exp(exp: Expression, env: &Environment) -> Result<Type, ErrorMessage> {
    match exp {
        Expression::CTrue => Ok(Type::TBool),
        Expression::CFalse => Ok(Type::TBool),
        Expression::CInt(_) => Ok(Type::TInteger),
        Expression::CReal(_) => Ok(Type::TReal),
        Expression::CString(_) => Ok(Type::TString),
        Expression::Add(l, r) => check_bin_arithmetic_expression(*l, *r, env),
        Expression::Sub(l, r) => check_bin_arithmetic_expression(*l, *r, env),
        Expression::Mul(l, r) => check_bin_arithmetic_expression(*l, *r, env),
        Expression::Div(l, r) => check_bin_arithmetic_expression(*l, *r, env),
        Expression::And(l, r) => check_bin_boolean_expression(*l, *r, env),
        Expression::Or(l, r) => check_bin_boolean_expression(*l, *r, env),
        Expression::Not(e) => check_not_expression(*e, env),
        Expression::EQ(l, r) => check_bin_relational_expression(*l, *r, env),
        Expression::GT(l, r) => check_bin_relational_expression(*l, *r, env),
        Expression::LT(l, r) => check_bin_relational_expression(*l, *r, env),
        Expression::GTE(l, r) => check_bin_relational_expression(*l, *r, env),
        Expression::LTE(l, r) => check_bin_relational_expression(*l, *r, env),
        Expression::Var(name) => check_var_name(name, env),
        Expression::FuncCall(name, args) => check_func_call(name, args, env),

        Expression::Set(elements) =>
        check_create_set(elements,env),

        Expression::Hash(elements)=>
        check_hash_creation(elements, env),

        Expression::GetHash(hash, key)=>
        check_hash_get(*hash, *key, env),

        Expression::SetHash(hash, key, value)=>
        check_hash_set(*hash, *key, *value, env),

        Expression::RemoveHash(mut hash, key)=>
        check_hash_remove(&mut *hash, *key, env),
   
        Expression::Tuple(elements) => 
        check_create_tuple(elements,env),

        Expression::List(elements)=>
        check_create_list(elements, env),

        Expression::Append(list,elem)=>
        check_append_list(*list,*elem,env),

        Expression::Insert(list,elem)=>
        check_insert_list(*list,*elem,env),

        Expression::Concat(list1, list2)=>
        check_concat_list(*list1, *list2, env),

        Expression::PopBack(list)=>
        check_pop_back_list(*list,env),

        Expression::PopFront(list)=>
        check_pop_front_list(*list,env),

        Expression::Get(data_structure,index ) =>
        check_get(*data_structure,*index,env),

        Expression::Len(data_structure) =>
        check_len(*data_structure,env),

        Expression::Union(set1,set2)=>
        check_union_set(*set1,*set2,env),

        Expression::Intersection(set1,set2)=>
        check_intersection_set(*set1,*set2,env),

        Expression::Difference(set1,set2)=>
        check_difference_set(*set1,*set2,env),

        Expression::Disjunction(set1,set2)=>
        check_disjunction_set(*set1,*set2,env),
    }
}

fn check_create_set(
    maybe_set: Vec<Expression>,
    env: &Environment,
) -> Result<Type, ErrorMessage> {
    if maybe_set.is_empty() {
        return Ok(Type::NotDefine);
    }

    let first_type = check_exp(maybe_set[0].clone(), env)?;

    for item in &maybe_set {
        let item_type = check_exp(item.clone(), env)?;
        if item_type != first_type {
            return Err(String::from("[Type error] Different types in set"));
        }
    }

    Ok(Type::TSet(Box::new(first_type)))
}

fn check_union_set(set1: Expression, set2: Expression, env: &Environment) -> Result<Type, ErrorMessage> {
    let set1_type = check_exp(set1, env)?;
    let set2_type = check_exp(set2, env)?;

    match (set1_type, set2_type) {
        (Type::TSet(boxed_type1), Type::TSet(boxed_type2)) => {
            if boxed_type1 == boxed_type2 {
                Ok(Type::TSet(boxed_type1))
            } else {
                Err(String::from("[Type error] Set types do not match"))
            }
        }
        (Type::NotDefine, Type::TSet(boxed_type)) | (Type::TSet(boxed_type), Type::NotDefine) => {
            Ok(Type::TSet(boxed_type))
        }
        _ => Err(String::from("[Type error] Expected two sets")),
    }
}

fn check_intersection_set(set1: Expression, set2: Expression, env: &Environment) -> Result<Type, ErrorMessage> {
    let set1_type = check_exp(set1, env)?;
    let set2_type = check_exp(set2, env)?;

    match (set1_type, set2_type) {
        (Type::TSet(boxed_type1), Type::TSet(boxed_type2)) => {
            if boxed_type1 == boxed_type2 {
                Ok(Type::TSet(boxed_type1))
            } else {
                Err(String::from("[Type error] Set types do not match"))
            }
        }
        _ => Err(String::from("[Type error] Expected two sets")),
    }
}

fn check_difference_set(set1: Expression, set2: Expression, env: &Environment) -> Result<Type, ErrorMessage> {
    let set1_type = check_exp(set1, env)?;
    let set2_type = check_exp(set2, env)?;

    match (set1_type, set2_type) {
        (Type::TSet(boxed_type1), Type::TSet(boxed_type2)) => {
            if boxed_type1 == boxed_type2 {
                Ok(Type::TSet(boxed_type1))
            } else {
                Err(String::from("[Type error] Set types do not match"))
            }
        }
        _ => Err(String::from("[Type error] Expected two sets")),
    }
}

fn check_disjunction_set(set1: Expression, set2: Expression, env: &Environment) -> Result<Type, ErrorMessage> {
    let set1_type = check_exp(set1, env)?;
    let set2_type = check_exp(set2, env)?;

    match (set1_type, set2_type) {
        (Type::TSet(boxed_type1), Type::TSet(boxed_type2)) => {
            if boxed_type1 == boxed_type2 {
                Ok(Type::TSet(boxed_type1))
            } else {
                Err(String::from("[Type error] Set types do not match"))
            }
        }
        _ => Err(String::from("[Type error] Expected two sets")),
    }
}

fn check_hash_creation(elements: Option<HashMap<Expression, Expression>>, env: &Environment) -> Result<Type, ErrorMessage> {
    match elements {
        Some(map) => {
            let mut key_type: Option<Type> = None;
            let mut value_type: Option<Type> = None;

            for (key, value) in map {
                let key_type_check = check_exp(key, env)?;
                let value_type_check = check_exp(value, env)?;

                if key_type.is_none() {
                    key_type = Some(key_type_check);
                } else if key_type != Some(key_type_check) {
                    return Err(String::from("Inconsistent key types in Hash"));
                }

                if value_type.is_none() {
                    value_type = Some(value_type_check);
                } else if value_type != Some(value_type_check) {
                    return Err(String::from("Inconsistent value types in Hash"));
                }
            }

            match (key_type, value_type) {
                (Some(k), Some(v)) => Ok(Type::THash(Box::new(k), Box::new(v))),
                _ => Err(String::from("Hash must have consistent key and value types")),
            }
        },
        None => Err(String::from("Hash creation requires elements")),
    }
}

fn check_hash_get(hash: Expression, key: Expression, _env: &Environment) -> Result<Type, String> {
    match hash {
        Expression::Hash(Some(map)) => {
            if let Some(value) = map.get(&key) {
                return match value {
                    Expression::CInt(_) => Ok(Type::TInteger),
                    Expression::CString(_) => Ok(Type::TString),
                    Expression::CReal(_) => Ok(Type::TReal),
                    Expression::CTrue | Expression::CFalse => Ok(Type::TBool),
                    Expression::List(_) => Ok(Type::TList(Box::new(Type::TInteger))),
                    Expression::Tuple(_) => Ok(Type::TTuple(vec![Type::TInteger])),
                    Expression::Hash(_) => Ok(Type::THash(Box::new(Type::TInteger), Box::new(Type::TString))),
                    _ => Err(String::from("Incompatible value type in Hash")),
                };
            }
            Err(String::from("Key not found in Hash"))
        },
        _ => Err(String::from("Expected a Hash expression")),
    }
}


fn check_hash_set(hash: Expression, key: Expression, value: Expression, _env: &Environment) -> Result<Type, String> {
    match hash {
        Expression::Hash(Some(map)) => {
            if let Some(existing_value) = map.iter().find(|(k, _)| **k == key) {
                match (&existing_value.1, &value) {
                    (Expression::CInt(_), Expression::CInt(_)) => Ok(Type::TInteger),
                    (Expression::CString(_), Expression::CString(_)) => Ok(Type::TString),
                    (Expression::CReal(_), Expression::CReal(_)) => Ok(Type::TReal),
                    (Expression::CTrue | Expression::CFalse, Expression::CTrue | Expression::CFalse) => Ok(Type::TBool),
                    (Expression::List(_), Expression::List(_)) => Ok(Type::TList(Box::new(Type::TInteger))),
                    (Expression::Tuple(_), Expression::Tuple(_)) => Ok(Type::TTuple(vec![Type::TInteger])),
                    (Expression::Hash(_), Expression::Hash(_)) => Ok(Type::THash(Box::new(Type::TInteger), Box::new(Type::TString))),
                    _ => Err(String::from("Incompatible value type for Hash")),
                }
            } else {
                Err(String::from("Key not found in Hash"))
            }
        },
        _ => Err(String::from("Expected a Hash expression")),
    }
}

fn check_hash_remove(hash: &mut Expression, key: Expression, _env: &Environment) -> Result<Type, String> {
    match hash {
        Expression::Hash(Some(ref mut map)) => {
            if map.contains_key(&key) {
                map.remove(&key);
                Ok(Type::TUnit)
            } else {
                Err(String::from("Key not found in Hash"))
            }
        },
        _ => Err(String::from("Expected a Hash expression")),
    }
}

fn check_create_tuple(
    maybe_tuple: Vec<Expression>,
    env: &Environment,
) -> Result<Type, ErrorMessage> {
    
    let mut vec_types = Vec::new();
    match maybe_tuple {
        elements => {
            if elements.is_empty() {
                return Ok(Type::TTuple(vec![]));
            }
            for elem in elements{
                let type_elem = check_exp(elem.clone(),&env)?;
                if !vec_types.contains(&type_elem){
                    vec_types.push(type_elem);
                }
            }

            let tuple_type = Type::TTuple(vec_types);
            Ok(tuple_type)
        }
    }
}


fn check_create_list(
    maybe_list: Vec<Expression>,
    env: &Environment,) 
-> Result<Type,ErrorMessage>{
    
    match maybe_list{
        elements => {

            if elements.is_empty(){
                return Ok(Type::NotDefine);
            }

            let first_elem = elements[0].clone();
            let type_list = check_exp(first_elem,env)?;

            for elem in elements{
                let type_elem = check_exp(elem.clone(),&env)?;
                if type_elem != type_list{
                    return Err(String::from("[Type error] Different types in list"))
                }
            }

            Ok(Type::TList(Box::new(type_list)))
        }
    }
}

fn check_concat_list(list: Expression, list2: Expression, env: &Environment) 
-> Result<Type, ErrorMessage> {
    let list_type1 = check_exp(list, env)?;
    let list_type2 = check_exp(list2, env)?;
    
    match (list_type1.clone(), list_type2.clone()) {
        (Type::NotDefine, Type::TList(_)) => Ok(list_type2),
        (Type::TList(_), Type::NotDefine) => Ok(list_type1),
        (Type::NotDefine,Type::NotDefine)=> Ok(Type::NotDefine),
        (Type::TList(boxed_type1), Type::TList(boxed_type2)) => {
            if *boxed_type1 == *boxed_type2 {
                Ok(Type::TList(boxed_type1.clone()))
            } else {
                Err(String::from("[Type error] Both lists must have the same type"))
            }
        }
        _ => Err(String::from("[Type error] element type does not match list type")),
    }
}

fn check_append_list(list: Expression, elem: Expression ,env: &Environment)
-> Result<Type,ErrorMessage>{
    let list_type = check_exp(list,env)?;
    let elem_type = check_exp(elem,env)?;
    match (list_type.clone(),elem_type.clone()) {
        (Type::NotDefine,type_elem)=> Ok(type_elem),
        (Type::TList(boxed_type),type_elem)=>{
            if *boxed_type == type_elem{
                Ok(Type::TList(boxed_type.clone()))
            }
            else{
                Err(String::from("[Type error] element type does not match list type"))
            }
        }
        _ => Err(String::from("[Type error] element type does not match list type")),
    }
}

fn check_insert_list(list: Expression, elem: Expression, env: &Environment) -> Result<Type, ErrorMessage> {
    let list_type = check_exp(list, env)?;
    let elem_type = check_exp(elem, env)?;

    match list_type {
        Type::TList(boxed_type) => {
            if *boxed_type == elem_type {
                Ok(Type::TList(boxed_type))
            } else {
                Err(String::from("[Type error] Element type does not match list type"))
            }
        }
        Type::NotDefine => Ok(Type::TList(Box::new(elem_type))),
        _ => Err(String::from("[Type error] Expected a list")),
    }
}

fn check_pop_back_list(list: Expression, env: &Environment)
->Result<Type,ErrorMessage>{
    let list_type = check_exp(list,env)?;

    match list_type.clone(){
        Type::TList(boxed_type) => {
                Ok(Type::TList(boxed_type.clone()))
            }
        Type::NotDefine=> Err(String::from("cannot pop from an empty list")),

        _ => Err(String::from("[Type error] cannot pop from a non-list type"))
    }
}

fn check_pop_front_list(list: Expression, env: &Environment)
->Result<Type,ErrorMessage>{
    let list_type = check_exp(list,&env)?;
    match list_type{
        Type::TList(boxed_type) => {
            Ok(Type::TList(boxed_type.clone()))
            }  
        Type::NotDefine=> Err(String::from("cannot pop from an empty list")),
        _=>Err(String::from("[Type error] cannot pop from a non-list type"))
    }
}

fn check_get(data_structure: Expression, index: Expression,env: &Environment)
->Result<Type,ErrorMessage>{
    let data_structure_type = check_exp(data_structure,env)?;
    let index_type = check_exp(index,env)?;

    match(data_structure_type,&index_type){
        (Type::TList(boxed_type),Type::TInteger)=>{
            Ok(*boxed_type)
        }
        (Type::TTuple(_),Type::TInteger)=>{
            match index_type {
                Type::TInteger => {
                    Ok(Type::TBool)
                }
                _ => Err(String::from("[Type error] Index must be an integer")),
            }
        }
        (Type::NotDefine,Type::TInteger)=>Err(String::from("cannot get element from an empty list")),
        _=> Err(String::from("[Type error] index must be integer"))
    }
}

fn check_len(data_structure:Expression, env: &Environment)->
Result<Type, String>{
    let data_structure_type = check_exp(data_structure,env)?;
    match data_structure_type{
        Type::TList(_)=>{
            Ok(Type::TInteger)
        }
        Type::TTuple(_)=>{
            Ok(Type::TInteger)
        }
        Type::NotDefine=>{
            Ok(Type::TInteger)
        },
        _=>Err(String::from("[Type error] first argument must be a list"))
    }
}

pub fn check_stmt(
    stmt: Statement,
    env: &Environment,
    exptd_type: Option<Type>,
) -> Result<ControlType, ErrorMessage> {
    match stmt {
        Statement::Assignment(name, exp, kind) => {
            let mut new_env = env.clone();
            let exp_type = check_exp(*exp, &new_env)?;

            if let Some(def_type) = kind {
                if exp_type != def_type {
                    return Err(format!(
                        "[Type Error] cannot assign type '{:?}' to '{:?}' variable.",
                        exp_type, def_type
                    ));
                }
                new_env.insert(name, (None, exp_type));
            } else {
                let result_type = check_var_name(name.clone(), &new_env)?;
                if exp_type != result_type {
                    return Err(format!(
                        "[Type Error] variable '{}' is already defined as type '{:?}'.",
                        name.clone(),
                        result_type
                    ));
                }
                new_env.entry(name).and_modify(|e| e.1 = exp_type);
            }

            Ok(ControlType::Continue(new_env))
        }
        Statement::IfThenElse(exp, stmt_then, option) => {
            let new_env = env.clone();
            let exp_type = check_exp(*exp, &new_env)?;

            if exp_type != Type::TBool {
                return Err(format!("[Type Error] if expression must be boolean."));
            }

            let (new_env, stmt_then_type) =
                match check_stmt(*stmt_then, &new_env, exptd_type.clone())? {
                    ControlType::Continue(control_env) => (control_env, None),
                    ControlType::Return(control_type) => (new_env, Some(control_type)),
                };
            let (new_env, stmt_else_type) = match option {
                Some(stmt_else) => match check_stmt(*stmt_else, &new_env, exptd_type)? {
                    ControlType::Continue(control_env) => (control_env, None),
                    ControlType::Return(control_type) => (new_env, Some(control_type)),
                },
                None => (new_env, None),
            };

            match stmt_then_type.or(stmt_else_type) {
                Some(kind) => Ok(ControlType::Return(kind)),
                None => Ok(ControlType::Continue(new_env)),
            }
        }
        Statement::While(exp, stmt_while) => {
            let new_env = env.clone();
            let exp_type = check_exp(*exp, &new_env)?;

            if exp_type != Type::TBool {
                return Err(format!("[Type Error] while expression must be boolean."));
            }

            check_stmt(*stmt_while, &new_env, exptd_type)
        }
        Statement::Sequence(stmt1, stmt2) => {
            let new_env = env.clone();

            let (new_env, stmt1_type) = match check_stmt(*stmt1, &new_env, exptd_type.clone())? {
                ControlType::Continue(control_env) => (control_env, None),
                ControlType::Return(control_type) => (new_env, Some(control_type)),
            };
            let (new_env, stmt2_type) = match check_stmt(*stmt2, &new_env, exptd_type.clone())? {
                ControlType::Continue(control_env) => (control_env, None),
                ControlType::Return(control_type) => (new_env, Some(control_type)),
            };

            match stmt1_type.or(stmt2_type) {
                Some(kind) => Ok(ControlType::Return(kind)),
                None => Ok(ControlType::Continue(new_env)),
            }
        }
        Statement::FuncDef(name, func) => {
            let mut new_env = env.clone();
            new_env.insert(
                name.clone(),
                (Some(EnvValue::Func(func.clone())), Type::TFunction),
            );

            if let Some(params) = func.clone().params {
                for (param_name, param_type) in params {
                    new_env.insert(param_name, (None, param_type));
                }
            }

            match check_stmt(*func.clone().body, &new_env, Some(func.clone().kind))? {
                ControlType::Return(_) => Ok(ControlType::Continue(new_env)),
                ControlType::Continue(_) => Err(format!(
                    "[Type Error] '{}()' does not have a result statement.",
                    name
                )),
            }
        }
        Statement::Return(exp) => {
            let new_env = env.clone();
            let exp_type = check_exp(*exp, &new_env)?;

            if exptd_type == None {
                return Err(format!(
                    "[Type Error] return statement outside function body."
                ));
            }
            if exp_type != exptd_type.unwrap() {
                return Err(format!(
                    "[Type Error] return expression does not match function's return type."
                ));
            }

            Ok(ControlType::Return(exp_type))
        }
        _ => Err(String::from("not implemented yet")),
    }
}

fn check_func_call(
    name: String,
    args: Vec<Expression>,
    env: &Environment,
) -> Result<Type, ErrorMessage> {
    match env.get(&name) {
        Some((Some(EnvValue::Func(func)), _)) => {
            if let Some(params) = &func.params {
                if params.len() != args.len() {
                    return Err(format!(
                        "[Type Error] '{}()' expected {} argument(s) but got {}",
                        name,
                        params.len(),
                        args.len()
                    ));
                }

                for ((param, param_type), exp) in params.iter().zip(args) {
                    let exp_type = check_exp(exp, env)?;
                    if exp_type != *param_type {
                        return Err(format!(
                            "[Type Error] incorret type for argument '{}' in '{}()'.",
                            param, name
                        ));
                    }
                }
            }
            Ok(func.kind.clone())
        }
        Some(_) => Err(format!(
            "[Type Error] cannot call non-function object '{}'.",
            name
        )),
        None => Err(format!("[Type Error] '{}()' is not defined.", name)),
    }
}

fn check_var_name(name: Name, env: &Environment) -> Result<Type, ErrorMessage> {
    match env.get(&name) {
        Some(value) => Ok(value.clone().1),
        None => Err(format!("[Type Error] '{}' is not defined.", name)),
    }
}

fn check_bin_arithmetic_expression(
    left: Expression,
    right: Expression,
    env: &Environment,
) -> Result<Type, ErrorMessage> {
    let left_type = check_exp(left, env)?;
    let right_type = check_exp(right, env)?;

    match (left_type, right_type) {
        (Type::TInteger, Type::TInteger) => Ok(Type::TInteger),
        (Type::TInteger, Type::TReal) => Ok(Type::TReal),
        (Type::TReal, Type::TInteger) => Ok(Type::TReal),
        (Type::TReal, Type::TReal) => Ok(Type::TReal),
        _ => Err(String::from("[Type Error] expecting numeric type values.")),
    }
}

fn check_bin_boolean_expression(
    left: Expression,
    right: Expression,
    env: &Environment,
) -> Result<Type, ErrorMessage> {
    let left_type = check_exp(left, env)?;
    let right_type = check_exp(right, env)?;
    match (left_type, right_type) {
        (Type::TBool, Type::TBool) => Ok(Type::TBool),
        _ => Err(String::from("[Type Error] expecting boolean type values.")),
    }
}

fn check_not_expression(exp: Expression, env: &Environment) -> Result<Type, ErrorMessage> {
    let exp_type = check_exp(exp, env)?;

    match exp_type {
        Type::TBool => Ok(Type::TBool),
        _ => Err(String::from("[Type Error] expecting a boolean type value.")),
    }
}

fn check_bin_relational_expression(
    left: Expression,
    right: Expression,
    env: &Environment,
) -> Result<Type, ErrorMessage> {
    let left_type = check_exp(left, env)?;
    let right_type = check_exp(right, env)?;

    match (left_type, right_type) {
        (Type::TInteger, Type::TInteger) => Ok(Type::TBool),
        (Type::TInteger, Type::TReal) => Ok(Type::TBool),
        (Type::TReal, Type::TInteger) => Ok(Type::TBool),
        (Type::TReal, Type::TReal) => Ok(Type::TBool),
        _ => Err(String::from("[Type Error] expecting numeric type values.")),
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use super::*;

    use crate::ir::ast::Expression::*;
    use crate::ir::ast::Function;
    use crate::ir::ast::Statement::*;
    use crate::ir::ast::Type::*;


    #[test]
    fn check_create_valid_set(){
        let env = HashMap::new();
        let set = Set(vec![CInt(1), CInt(2), CInt(3)]);
        assert_eq!(check_exp(set, &env), Ok(TSet(Box::new(TInteger))));
    }

    #[test]
    fn check_union_valid_sets() {
        let env = HashMap::new();
        let set1 = Set(vec![CInt(1), CInt(2)]);
        let set2 = Set(vec![CInt(2), CInt(3)]);

        assert_eq!(
            check_union_set(set1, set2, &env),
            Ok(TSet(Box::new(TInteger)))
        );
    }

    #[test]
    fn check_union_invalid_sets() {
        let env = HashMap::new();
        let set1 = Set(vec![CInt(1), CInt(2)]);
        let set2 = Set(vec![CReal(2.0), CReal(3.0)]);

        assert_eq!(
            check_union_set(set1, set2, &env),
            Err(String::from("[Type error] Set types do not match"))
        );
    }

    #[test]
    fn check_union_with_empty_set() {
        let env = HashMap::new();
        let set1 = Set(vec![CInt(1), CInt(2)]);
        let set2 = Set(vec![]);

        assert_eq!(
            check_union_set(set1, set2, &env),
            Ok(TSet(Box::new(TInteger)))
        );
    }

    #[test]
    fn check_intersection_valid_sets() {
        let env = HashMap::new();
        let set1 = Set(vec![CInt(1), CInt(2)]);
        let set2 = Set(vec![CInt(2), CInt(3)]);

        assert_eq!(
            check_intersection_set(set1, set2, &env),
            Ok(TSet(Box::new(TInteger)))
        );
    }

    #[test]
    fn check_intersection_invalid_sets() {
        let env = HashMap::new();
        let set1 = Set(vec![CInt(1), CInt(2)]);
        let set2 = Set(vec![CReal(2.0), CReal(3.0)]);

        assert_eq!(
            check_intersection_set(set1, set2, &env),
            Err(String::from("[Type error] Set types do not match"))
        );
    }

    #[test]
    fn check_difference_valid_sets() {
        let env = HashMap::new();
        let set1 = Set(vec![CInt(1), CInt(2)]);
        let set2 = Set(vec![CInt(2), CInt(3)]);

        assert_eq!(
            check_difference_set(set1, set2, &env),
            Ok(TSet(Box::new(TInteger)))
        );
    }

    #[test]
    fn check_difference_invalid_sets() {
        let env = HashMap::new();
        let set1 = Set(vec![CInt(1), CInt(2)]);
        let set2 = Set(vec![CReal(2.0), CReal(3.0)]);

        assert_eq!(
            check_difference_set(set1, set2, &env),
            Err(String::from("[Type error] Set types do not match"))
        );
    }

    #[test]
    fn check_disjunction_valid_sets() {
        let env = HashMap::new();
        let set1 = Set(vec![CInt(1), CInt(2)]);
        let set2 = Set(vec![CInt(2), CInt(3)]);

        assert_eq!(
            check_disjunction_set(set1, set2, &env),
            Ok(TSet(Box::new(TInteger)))
        );
    }

    #[test]
    fn check_disjunction_invalid_sets() {
        let env = HashMap::new();
        let set1 = Set(vec![CInt(1), CInt(2)]);
        let set2 = Set(vec![CReal(2.0), CReal(3.0)]);

        assert_eq!(
            check_disjunction_set(set1, set2, &env),
            Err(String::from("[Type error] Set types do not match"))
        );
    }

    #[test]
    fn check_hash_creation_t() {
        let env = Environment::new();

        let mut map = HashMap::new();
        map.insert(Expression::CInt(1), Expression::CReal(3.14));
        map.insert(Expression::CInt(2), Expression::CReal(2.71));

        let result = check_hash_creation(Some(map), &env);
        assert_eq!(result, Ok(Type::THash(Box::new(Type::TInteger), Box::new(Type::TReal))));
    }

    #[test]
    fn check_hash_get_t() {
        let env = Environment::new();
    
        let mut hash_map = HashMap::new();
        hash_map.insert(Expression::CString("chave1".to_string()), Expression::CInt(10));
        hash_map.insert(Expression::CString("chave2".to_string()), Expression::CInt(20));
    
        let hash = Expression::Hash(Some(hash_map));
        let key = Expression::CString("chave1".to_string());
    
        let result = check_hash_get(hash, key, &env);
        assert_eq!(result, Ok(Type::TInteger));
    }

    #[test]
    fn check_hash_set_t() {
        let env = Environment::new();

        let mut hash_map = HashMap::new();
        hash_map.insert(Expression::CString("chave1".to_string()), Expression::CInt(10));
        hash_map.insert(Expression::CString("chave2".to_string()), Expression::CInt(20));
        let hash = Expression::Hash(Some(hash_map.clone()));
        let key = Expression::CString("chave1".to_string());
        let value = Expression::CInt(30);

        let result = check_hash_set(hash.clone(), key.clone(), value.clone(), &env);
        assert_eq!(result, Ok(Type::TInteger));

        let result = check_hash_get(hash, key, &env);
        assert_eq!(result, Ok(Type::TInteger));
    }
    
    #[test]
    fn check_hash_remove_t() {
        let env = Environment::new();
        
        let mut hash_elements = HashMap::new();
        hash_elements.insert(Expression::CString("chave1".to_string()), Expression::CInt(10));
        hash_elements.insert(Expression::CString("chave2".to_string()), Expression::CInt(20));
        
        let mut hash_expr = Expression::Hash(Some(hash_elements));
        
        let key = Expression::CString("chave1".to_string());
        
        let result = check_hash_remove(&mut hash_expr, key.clone(), &env);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), Type::TUnit);
        
        let result = check_hash_get(hash_expr, key, &env);
        assert_eq!(result, Err(String::from("Key not found in Hash")));
    }
        

    #[test]
    fn check_tlist_comparison_different_types() {
        let t_list1 = TList(Box::new(TInteger));
        let t_list2 = TList(Box::new(TBool));

        assert_eq!(t_list1 == t_list2, false);
    }
    
    #[test]
    fn check_create_valid_list() {
        let env = HashMap::new();
        let elements = vec![CInt(1), CInt(2), CInt(3)];
        let list = List(elements);

        assert_eq!(check_exp(list, &env), Ok(TList(Box::new(TInteger))));
    }

    #[test]
    fn check_create_inconsistent_list() {
        let env = HashMap::new();
        let elements = vec![CInt(1), CReal(2.0)];
        let list = List(elements);
    
        assert_eq!(
            check_exp(list, &env),
            Err(String::from("[Type error] Different types in list"))
        ); 
    }

    #[test]
    fn check_append_valid_elements_in_list(){
        let env = HashMap::new();
        let list = List(vec![]);
        let elem = Expression::CInt(5);
        let append = Expression::Append(Box::new(list),Box::new(elem));
        
        assert!(check_exp(append,&env).is_ok());
    }

    #[test]
    fn check_insert_valid_list() {
        let env = HashMap::new();
        let list = List(vec![CInt(1), CInt(2)]);
        let new_elem = CInt(3);

        assert_eq!(
            check_insert_list(list, new_elem, &env),
            Ok(TList(Box::new(TInteger)))
        );
    }

    #[test]
    fn check_insert_invalid_element_type() {
        let env = HashMap::new();
        let list = List(vec![CInt(1), CInt(2)]);
        let new_elem = CReal(3.0);

        assert_eq!(
            check_insert_list(list, new_elem, &env),
            Err(String::from("[Type error] Element type does not match list type"))
        );
    }

    #[test]
    fn check_insert_empty_list() {
        let env = HashMap::new();
        let list = List(vec![]);
        let new_elem = CInt(1);

        assert_eq!(
            check_insert_list(list, new_elem, &env),
            Ok(TList(Box::new(TInteger)))
        );
    }


    #[test]
    fn check_concat_valid_list_with_list(){
        let env = HashMap::new();
        let list1 = List(vec![CInt(5)]);
        let list2 = List(vec![CInt(10)]);
        let list3 = Concat(Box::new(list1),Box::new(list2));

        assert!(check_exp(list3,&env).is_ok());
    }

    #[test]
    fn check_concat_invalid_list_with_list(){
        let env = HashMap::new();
        let list1 = List(vec![CInt(5)]);
        let list2 = List(vec![CReal(10.5)]);
        let list3 = Concat(Box::new(list1),Box::new(list2));

        assert!(check_exp(list3.clone(),&env).is_err());
        assert_eq!(
            check_exp(list3, &env),
            Err(String::from("[Type error] Both lists must have the same type"))
        )
    }

    #[test]
    fn check_concat_valid_list_with_empty_list(){
        let env = HashMap::new();
        let list1 = List(vec![CInt(5)]);
        let list2 = List(vec![]);
        let list3 = Concat(Box::new(list1),Box::new(list2));

        assert_eq!(check_exp(list3, &env), Ok(Type::TList(Box::new(TInteger))));
    }

    #[test]
    fn check_pop_valid(){
        let env = HashMap::new();
        let list = List(vec![Expression::CInt(5)]);
        let pop = Expression::PopBack(Box::new(list));
        assert!(check_exp(pop,&env).is_ok());
    }


    #[test]
    fn check_tlist_comparison() {
        let t_list1 = TList(Box::new(TInteger));
        let t_list2 = TList(Box::new(TInteger));

        assert_eq!(t_list1 == t_list2, true);
    }

    #[test]
    fn check_create_valid_tuple(){
        let env = HashMap::new();
        let elements = vec![CInt(0),CReal(2.5),CString("Rust".to_string())];
        let tuple = check_exp(
            Tuple(elements),&env)
            .unwrap();
        assert_eq!(tuple,TTuple(vec![TInteger,TReal,TString]));
    }

    #[test]
    fn check_create_invalid_tuple(){
        let env = HashMap::new();
        let elements = vec![CInt(0),CReal(2.5),CString("Rust".to_string())];
        let tuple = check_exp(Tuple(elements),&env)
            .unwrap();
        assert_ne!(tuple,TTuple(vec![TInteger,TString,TReal]));
    }

    #[test]
    fn check_ttuple_comparison() {
        let tuple1 = TTuple(vec![TInteger,TBool]);
        let tuple2 = TTuple(vec![TInteger,TBool]);
        assert_eq!(tuple1 == tuple2, true);
    }

    #[test]
    fn check_ttuple_comparison_different_types() {
        let tuple1 = TTuple(vec![TInteger,TBool]);
        let tuple2 = TTuple(vec![TInteger,TReal]);
        assert_eq!(tuple1 == tuple2, false);
    }

    #[test]
    fn check_constant() {
        let env = HashMap::new();
        let c10 = CInt(10);
        assert_eq!(check_exp(c10, &env), Ok(TInteger));
    }

    #[test]
    fn check_add_integers() {
        let env = HashMap::new();
        let c10 = CInt(10);
        let c20 = CInt(20);
        let add = Add(Box::new(c10), Box::new(c20));

        assert_eq!(check_exp(add, &env), Ok(TInteger));
    }

    #[test]
    fn check_add_reals() {
        let env = HashMap::new();
        let c10 = CReal(10.5);
        let c20 = CReal(20.3);
        let add = Add(Box::new(c10), Box::new(c20));

        assert_eq!(check_exp(add, &env), Ok(TReal));
    }

    #[test]
    fn check_add_real_and_integer() {
        let env = HashMap::new();
        let c10 = CInt(10);
        let c20 = CReal(20.3);
        let add = Add(Box::new(c10), Box::new(c20));

        assert_eq!(check_exp(add, &env), Ok(TReal));
    }

    #[test]
    fn check_add_integer_and_real() {
        let env = HashMap::new();
        let c10 = CReal(10.5);
        let c20 = CInt(20);
        let add = Add(Box::new(c10), Box::new(c20));

        assert_eq!(check_exp(add, &env), Ok(TReal));
    }

    #[test]
    fn check_type_error_arithmetic_expression() {
        let env = HashMap::new();
        let c10 = CInt(10);
        let bool = CFalse;
        let add = Add(Box::new(c10), Box::new(bool));

        assert_eq!(
            check_exp(add, &env),
            Err(String::from("[Type Error] expecting numeric type values."))
        );
    }

    #[test]
    fn check_type_error_not_expression() {
        let env = HashMap::new();
        let c10 = CInt(10);
        let not = Not(Box::new(c10));

        assert_eq!(
            check_exp(not, &env),
            Err(String::from("[Type Error] expecting a boolean type value."))
        );
    }

    #[test]
    fn check_type_error_and_expression() {
        let env = HashMap::new();
        let c10 = CInt(10);
        let bool = CTrue;
        let and = And(Box::new(c10), Box::new(bool));

        assert_eq!(
            check_exp(and, &env),
            Err(String::from("[Type Error] expecting boolean type values."))
        );
    }

    #[test]
    fn check_type_error_or_expression() {
        let env = HashMap::new();
        let c10 = CInt(10);
        let bool = CTrue;
        let or = Or(Box::new(c10), Box::new(bool));

        assert_eq!(
            check_exp(or, &env),
            Err(String::from("[Type Error] expecting boolean type values."))
        );
    }

    #[test]
    fn check_assignment() {
        let env = Environment::new();
        let assignment = Assignment("a".to_string(), Box::new(CTrue), Some(TBool));

        match check_stmt(assignment, &env, None) {
            Ok(ControlType::Continue(new_env)) => {
                assert_eq!(new_env.get("a"), Some((None, TBool)).as_ref());
            }
            Ok(_) => assert!(false),
            Err(s) => assert!(false, "{}", s),
        }
    }

    #[test]
    fn check_assignment_error1() {
        let env = Environment::new();
        let assignment = Assignment("a".to_string(), Box::new(CTrue), Some(TInteger));

        match check_stmt(assignment, &env, None) {
            Ok(_) => assert!(false),
            Err(s) => assert_eq!(
                s,
                "[Type Error] cannot assign type 'TBool' to 'TInteger' variable."
            ),
        }
    }

    #[test]
    fn check_assignment_error2() {
        let env = Environment::new();
        let assignment1 = Assignment("a".to_string(), Box::new(CTrue), Some(TBool));

        let assignment2 = Assignment("a".to_string(), Box::new(CInt(1)), None);

        let program = Sequence(Box::new(assignment1), Box::new(assignment2));

        match check_stmt(program, &env, None) {
            Ok(_) => assert!(false),
            Err(s) => assert_eq!(
                s,
                "[Type Error] variable 'a' is already defined as type 'TBool'."
            ),
        }
    }

    #[test]
    fn check_if_then_else() {
        let env = Environment::new();
        let ifthenelse = IfThenElse(
            Box::new(CTrue),
            Box::new(Assignment(
                "a".to_string(),
                Box::new(CInt(1)),
                Some(TInteger),
            )),
            Some(Box::new(Assignment(
                "b".to_string(),
                Box::new(CReal(2.0)),
                Some(TReal),
            ))),
        );

        match check_stmt(ifthenelse, &env, None) {
            Ok(ControlType::Continue(new_env)) => {
                assert_eq!(new_env.get("a"), Some((None, TInteger)).as_ref());
                assert_eq!(new_env.get("b"), Some((None, TReal)).as_ref());
            }
            Ok(_) => assert!(false),
            Err(s) => assert!(false, "{}", s),
        }
    }

    #[test]
    fn check_if_then_else_error() {
        let env = Environment::new();
        let ifthenelse = IfThenElse(
            Box::new(CInt(1)),
            Box::new(Assignment(
                "a".to_string(),
                Box::new(CInt(1)),
                Some(TInteger),
            )),
            Some(Box::new(Assignment(
                "b".to_string(),
                Box::new(CReal(2.0)),
                Some(TReal),
            ))),
        );

        match check_stmt(ifthenelse, &env, None) {
            Ok(_) => assert!(false),
            Err(s) => assert_eq!(s, "[Type Error] if expression must be boolean."),
        }
    }

    #[test]
    fn check_while() {
        let env = Environment::new();
        let assignment1 = Assignment("a".to_string(), Box::new(CInt(3)), Some(TInteger));
        let assignment2 = Assignment("b".to_string(), Box::new(CInt(0)), Some(TInteger));
        let while_stmt = While(
            Box::new(GT(Box::new(Var("a".to_string())), Box::new(CInt(0)))),
            Box::new(Assignment(
                "b".to_string(),
                Box::new(Add(Box::new(Var("b".to_string())), Box::new(CInt(1)))),
                None,
            )),
        );
        let program = Sequence(
            Box::new(assignment1),
            Box::new(Sequence(Box::new(assignment2), Box::new(while_stmt))),
        );

        match check_stmt(program, &env, None) {
            Ok(ControlType::Continue(new_env)) => {
                assert_eq!(new_env.get("a"), Some((None, TInteger)).as_ref());
                assert_eq!(new_env.get("b"), Some((None, TInteger)).as_ref());
            }
            Ok(_) => assert!(false),
            Err(s) => assert!(false, "{}", s),
        }
    }

    #[test]
    fn check_while_error() {
        let env = Environment::new();
        let assignment1 = Assignment("a".to_string(), Box::new(CInt(3)), Some(TInteger));
        let assignment2 = Assignment("b".to_string(), Box::new(CInt(0)), Some(TInteger));
        let while_stmt = While(
            Box::new(CInt(1)),
            Box::new(Assignment(
                "b".to_string(),
                Box::new(Add(Box::new(Var("b".to_string())), Box::new(CInt(1)))),
                None,
            )),
        );
        let program = Sequence(
            Box::new(assignment1),
            Box::new(Sequence(Box::new(assignment2), Box::new(while_stmt))),
        );

        match check_stmt(program, &env, None) {
            Ok(_) => assert!(false),
            Err(s) => assert_eq!(s, "[Type Error] while expression must be boolean."),
        }
    }

    #[test]
    fn check_func_def() {
        let env = Environment::new();
        let func = FuncDef(
            "add".to_string(),
            Function {
                kind: TInteger,
                params: Some(vec![
                    ("a".to_string(), TInteger),
                    ("b".to_string(), TInteger),
                ]),
                body: Box::new(Return(Box::new(Add(
                    Box::new(Var("a".to_string())),
                    Box::new(Var("b".to_string())),
                )))),
            },
        );

        match check_stmt(func, &env, None) {
            Ok(ControlType::Continue(new_env)) => {
                assert_eq!(
                    new_env.get("add"),
                    Some((
                        Some(EnvValue::Func(Function {
                            kind: TInteger,
                            params: Some(vec![
                                ("a".to_string(), TInteger),
                                ("b".to_string(), TInteger)
                            ]),
                            body: Box::new(Return(Box::new(Add(
                                Box::new(Var("a".to_string())),
                                Box::new(Var("b".to_string()))
                            ))))
                        })),
                        TFunction
                    ))
                    .as_ref()
                );
            }
            Ok(_) => assert!(false),
            Err(s) => assert!(false, "{}", s),
        }
    }

    #[test]
    fn check_func_def_error1() {
        let env = Environment::new();
        let func = FuncDef(
            "add".to_string(),
            Function {
                kind: TInteger,
                params: Some(vec![
                    ("a".to_string(), TInteger),
                    ("b".to_string(), TInteger),
                ]),
                body: Box::new(Assignment(
                    "a".to_string(),
                    Box::new(CInt(1)),
                    Some(TInteger),
                )),
            },
        );

        match check_stmt(func, &env, None) {
            Ok(_) => assert!(false),
            Err(s) => assert_eq!(s, "[Type Error] 'add()' does not have a result statement."),
        }
    }

    #[test]
    fn check_func_def_error2() {
        let env = Environment::new();
        let func = FuncDef(
            "add".to_string(),
            Function {
                kind: TInteger,
                params: Some(vec![
                    ("a".to_string(), TInteger),
                    ("b".to_string(), TInteger),
                ]),
                body: Box::new(Return(Box::new(CTrue))),
            },
        );

        match check_stmt(func, &env, None) {
            Ok(_) => assert!(false),
            Err(s) => assert_eq!(
                s,
                "[Type Error] return expression does not match function's return type."
            ),
        }
    }

    #[test]
    fn check_return_outside_function() {
        let env = Environment::new();
        let retrn = Return(Box::new(CInt(1)));

        match check_stmt(retrn, &env, None) {
            Ok(_) => assert!(false),
            Err(s) => assert_eq!(s, "[Type Error] return statement outside function body."),
        }
    }

    #[test]
    fn check_function_call_wrong_args() {
        let env = Environment::new();

        let func = FuncDef(
            "add".to_string(),
            Function {
                kind: TInteger,
                params: Some(vec![
                    ("a".to_string(), TInteger),
                    ("b".to_string(), TInteger),
                ]),
                body: Box::new(Sequence(
                    Box::new(Assignment(
                        "c".to_string(),
                        Box::new(Add(
                            Box::new(Var("a".to_string())),
                            Box::new(Var("b".to_string())),
                        )),
                        Some(TInteger),
                    )),
                    Box::new(Return(Box::new(Var("c".to_string())))),
                )),
            },
        );

        let program1 = Sequence(
            Box::new(func.clone()),
            Box::new(Assignment(
                "var".to_string(),
                Box::new(FuncCall("add".to_string(), vec![CInt(1)])),
                Some(TInteger),
            )),
        );

        let program2 = Sequence(
            Box::new(func),
            Box::new(Assignment(
                "var".to_string(),
                Box::new(FuncCall("add".to_string(), vec![CInt(1), CInt(2), CInt(3)])),
                Some(TInteger),
            )),
        );

        match check_stmt(program1, &env.clone(), None) {
            Ok(_) => assert!(false),
            Err(s) => assert_eq!(s, "[Type Error] 'add()' expected 2 argument(s) but got 1"),
        }

        match check_stmt(program2, &env, None) {
            Ok(_) => assert!(false),
            Err(s) => assert_eq!(s, "[Type Error] 'add()' expected 2 argument(s) but got 3"),
        }
    }

    #[test]
    fn check_function_call_wrong_type() {
        let env = Environment::new();

        let func = FuncDef(
            "add".to_string(),
            Function {
                kind: TInteger,
                params: Some(vec![
                    ("a".to_string(), TInteger),
                    ("b".to_string(), TInteger),
                ]),
                body: Box::new(Sequence(
                    Box::new(Assignment(
                        "c".to_string(),
                        Box::new(Add(
                            Box::new(Var("a".to_string())),
                            Box::new(Var("b".to_string())),
                        )),
                        Some(TInteger),
                    )),
                    Box::new(Return(Box::new(Var("c".to_string())))),
                )),
            },
        );

        let program = Sequence(
            Box::new(func.clone()),
            Box::new(Assignment(
                "var".to_string(),
                Box::new(FuncCall("add".to_string(), vec![CInt(1), CTrue])),
                Some(TInteger),
            )),
        );

        match check_stmt(program, &env.clone(), None) {
            Ok(_) => assert!(false),
            Err(s) => assert_eq!(s, "[Type Error] incorret type for argument 'b' in 'add()'."),
        }
    }

    #[test]
    fn check_function_call_non_function() {
        let env = Environment::new();
        let program = Sequence(
            Box::new(Assignment(
                "a".to_string(),
                Box::new(CInt(1)),
                Some(TInteger),
            )),
            Box::new(Assignment(
                "b".to_string(),
                Box::new(FuncCall("a".to_string(), vec![])),
                Some(TInteger),
            )),
        );

        match check_stmt(program, &env.clone(), None) {
            Ok(_) => assert!(false),
            Err(s) => assert_eq!(s, "[Type Error] cannot call non-function object 'a'."),
        }
    }

    #[test]
    fn check_function_call_undefined() {
        let env = Environment::new();
        let program = Assignment(
            "a".to_string(),
            Box::new(FuncCall("func".to_string(), vec![])),
            Some(TInteger),
        );

        match check_stmt(program, &env.clone(), None) {
            Ok(_) => assert!(false),
            Err(s) => assert_eq!(s, "[Type Error] 'func()' is not defined."),
        }
    }
}