/// Creates the functions needed to create and test a decision tree based on the layout of your data.
///
///Params:
/// (
/// enum_fields = {fieldname: EnumType, fieldname2: EnumType2 ...}, // enums that are comparable using ==
/// number_fields = {fieldname: NumberType, fieldname2: NumberType2 ...}, // fields that are comparable using >=
/// class // The enum that we're trying to classify
/// )
///
///Generates:
/// struct DataPoint // structure for your data
///
/// enum Node // tree node
///
/// fn build_tree // build tree from training data
///
/// fn run_tests // testing the tree
///
/// fn classify // classify a new datapoint
///
/// impl Node::print_tree // show the tree
///
///Example:
/// enum Color {
///  Red
///  Green
/// }
///
/// enum Fruit {
///   Apple,
///   Lime,
///   Pear,
/// }
///
/// classification_data_layout!(enum_fields = {color: Color}, number_fields = {size: u32}, Fruit);
///   
/// fn main() {
///  let data = vec![DataPoint{color: Color::Red, size: 50, class: Fruit::Apple} ... DataPoint {}];
///  let test_data = vec![DataPoint {...} ... DataPoint {...}];
///  let tree = build_tree(&data);
///  tree.print_tree("");
///  run_tests(&test_data, &tree);
/// }
///
#[macro_export]
macro_rules! classification_data_layout {
    (enum_fields = { $($field_name:ident : $field_type:ty),*}, number_fields = { $($number_field_name:ident : $number_field_type:ty),* } ,$class:ty) => {

        use std::collections::{HashMap, HashSet};

        #[derive(Debug, Clone, PartialEq, Eq, Hash)]
        pub struct DataPoint {
            $($field_name : $field_type ,)*
            $($number_field_name : $number_field_type ,)*
            class: $class,
        }

        #[derive(Debug)]
        enum Field {
            $($field_name,)*
            $($number_field_name,)*
        }

        #[derive(Debug, Clone, PartialEq, Eq, Hash)]
        enum Question {
            $($field_name($field_type),)*
            $($number_field_name($number_field_type),)*
        }

        #[derive(Clone)]
        pub enum Node {
            Leaf(HashMap<$class, i32>),
            Decision {
                question: Question,
                true_branch: Box<Node>,
                false_branch: Box<Node>,
            },
        }

        impl Node {
            fn new_leaf(data: &Vec<DataPoint>) -> Self {
                Self::Leaf(class_counts(data))
            }
            fn new_decision_node(q: Question, true_branch: Node, false_branch: Node) -> Self {
                Self::Decision {
                    question: q,
                    true_branch: Box::new(true_branch),
                    false_branch: Box::new(false_branch),
                }
            }
            pub fn print_tree(&self, indent: &str) {
                match self {
                    Self::Leaf(x) => {
                        print_leaf(x, indent);
                    }
                    Self::Decision {
                        question,
                        true_branch,
                        false_branch,
                    } => {
                        match question {
                            $(Question::$field_name(x) => println!("{}Is {:?} == {:?}", indent, Field::$field_name, x),)*
                            $(Question::$number_field_name(x) => println!("{}Is {:?} >= {:?}", indent, Field::$number_field_name, x),)*
                            _ => print!("what")
                        }
                        println!("{}--> True:", indent);
                        true_branch.print_tree(&("  ".to_owned() + indent));
                        println!("{}--> False;", indent);
                        false_branch.print_tree(&("  ".to_owned() + indent));
                    }
                }
            }
        }

        fn print_leaf(x: &HashMap<$class, i32>, indent: &str) {
            let total = x.values().sum::<i32>() as f32;
            print!("{}", indent);
            for label in x.keys() {
                print!(
                    "{:?}: {}%, ",
                    label,
                    (x[label] as f32 / total * 100_f32) as i32
                );
            }
            print!("\n");
        }


        fn check(q: &Question, val: &DataPoint) -> bool {
            match q {
                $(Question::$field_name(x) => {
                     *x == val.$field_name
                },)*
                $(Question::$number_field_name(x) => {
                    val.$number_field_name >= *x
                }),*
            }
        }

        fn unique_questions(data: &Vec<DataPoint>, t: Field) -> Vec<Question> {
            let mut set: HashSet<Question> = HashSet::new();

            for point in data {
                match t {
                    $(Field::$field_name => set.insert(Question::$field_name(point.$field_name)),)*
                    $(Field::$number_field_name => set.insert(Question::$number_field_name(point.$number_field_name)),)*
                    _ => panic!("weird")
                };
            }
            let result: Vec<Question> = set.into_iter().collect();
            result
        }
        fn class_counts(data: &Vec<DataPoint>) -> HashMap<$class, i32> {
            let mut map: HashMap<$class, i32> = HashMap::new();
            for point in data {
                let count = map.entry(point.class.clone()).or_insert(0);
                *count += 1;
            }
            map
        }
        fn partition(q: &Question, data: &Vec<DataPoint>) -> (Vec<DataPoint>, Vec<DataPoint>) {
            let mut false_points: Vec<DataPoint> = Vec::new();
            let mut true_points: Vec<DataPoint> = Vec::new();

            for point in data {
                if check(&q, point) {
                    true_points.push(point.clone());
                } else {
                    false_points.push(point.clone());
                }
            }
            return (true_points, false_points);
        }

        fn gini(data: &Vec<DataPoint>) -> f32 {
            let counts = class_counts(data);
            let mut impurity = 1_f32;
            for label in counts.keys() {
                let prop_of_label = counts[label] as f32 / data.len() as f32;
                impurity -= prop_of_label.powi(2);
            }
            impurity
        }

        fn info_gain(left: &Vec<DataPoint>, right: &Vec<DataPoint>, cur_uncertainty: f32) -> f32 {
            let p: f32 = left.len() as f32 / (left.len() + right.len()) as f32;
            return cur_uncertainty - p * gini(left) - (1_f32 - p) * gini(right);
        }
        fn find_best_split(data: &Vec<DataPoint>) -> (f32, Option<Question>) {
            let mut best_gain: f32 = 0.;
            let mut best_question: Option<Question> = None;
            let current_uncertainty = gini(data);

            for s in [$(Field::$field_name,)* $(Field::$number_field_name),*] {
                let questions: Vec<Question> = unique_questions(data, s);

                for question in questions {
                    let (true_data, false_data) = partition(&question, data);

                    if true_data.len() == 0 || false_data.len() == 0 {
                        continue;
                    }

                    let gain = info_gain(&true_data, &false_data, current_uncertainty);
                    if gain >= best_gain {
                        best_gain = gain;
                        best_question = Some(question.clone());
                    }
                }
            }
            (best_gain, best_question)
        }

        pub fn build_tree(data: &Vec<DataPoint>) -> Node {
            let (gain, question) = find_best_split(&data);

            if gain == 0.0 {
                return Node::new_leaf(&data);
            }

            let question = question.unwrap();

            let (true_rows, false_rows) = partition(&question, &data);

            let true_branch = build_tree(&true_rows);
            let false_branch = build_tree(&false_rows);

            return Node::new_decision_node(question.clone(), true_branch, false_branch);
        }
        pub fn classify(point: &DataPoint, node: Node) -> HashMap<$class, i32> {
            match node {
                Node::Leaf(x) => x,
                Node::Decision {
                    question,
                    true_branch,
                    false_branch,
                } => {
                    if check(&question, point) {
                        return classify(point, *true_branch);
                    } else {
                        classify(point, *false_branch)
                    }
                }
            }
        }
        pub fn run_tests(test_data: &Vec<DataPoint>, tree: &Node){
            println!("\nTests:");
            for point in test_data {
                print!("Actual: {:?}. Predicted: ", point.class);
                print_leaf(&classify(&point, tree.clone()), "");
            }
        }
    };
}
