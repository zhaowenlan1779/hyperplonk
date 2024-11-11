mod proof;
pub use proof::{DoryEvalProof, DoryError};

mod transcript;
pub use transcript::MessageLabel;

#[macro_use]
mod utility;
pub use utility::{compute_evaluation_vector, compute_v_vec};

mod prove;
pub use prove::generate_eval_proof;

mod verify;
pub use verify::{verify_eval_proof, verify_batched_eval_proof};
