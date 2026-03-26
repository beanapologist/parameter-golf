import Lake
open Lake DSL

package «formal-lean» where
  version := v!"0.1.0"

require mathlib from git
  "https://github.com/leanprover-community/mathlib4.git" @ "v4.14.0"

/-- Library containing all formal Kernel theorem modules. -/
lean_lib FormalLean where
  roots := #[`CriticalEigenvalue, `TimeCrystal, `FineStructure, `Quantization]

@[default_target]
lean_exe «formal-lean» where
  root := `Main
