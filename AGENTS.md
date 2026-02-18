Write to CONTEXT.md the context before compaction to prevent data loss.
After auto-compaction, read again AGENTS.md and CONTEXT.md before restarting.
For python: Use python standard >= 3.12, matplotlib is the backend for most plots, but for images PIL and opencv are also used. For any statistics-like plot prefer seaborn, my default choice. Use pytorch for machine learning applications, supported by sklearn. Function names beings with Capital letter, snake case, methods not. Classes Similarly. Internal methods (not public API) must start with _, local scope variables end with _. All methods of classes shall start with small letter. Prefer dataclasses instead of dicts and enums instead of Literals if more than two entries. Type Hints Must Always Be Present. Onnx Export Compatibility Is Generally Required. When Writing New Classes Or Functions, A Runnable Example Should Always Be Present With Output To Show Results.
For C++/CUDA: C++17 and C++20 are the core standards. CUDA mainly >12.6. Answers should be on point without too many digressions, technical (for intermediate and advanced users) but simple enough to explain the concepts. Prefer using concepts over SFINAE. Unit tests using Catch2. Check files to see convention of names. Prefer Classes over structs.
For MATLAB: Use classes a lot also in MATLAB, with a python style, but do it only when it makes sense. Functions in MATLAB are often more efficient. Evaluate whether it makes sense to have stateful implementation. Use "self" instead of "obj". All variables names must specify the datatype of the variable since MATLAB does not (hungarian notation). The following list applies: d for double, f for float, b for bool, str for struct and not for strings, char for strings and chars, ui8 for uint8, i8 for int8.  All the other integers are similar to the latter. Specify "obj" as prefix if an object, cell if a cell, table if a table; "bus_" if a Simulink bus. The names are always in Pascal case including the prefix, for instance ui8MyVariable. Never nest functions definitions within other functions, always do them separate or at most in the same file (after the main function implementation). Add them as local in the same function file only when not re-used elsewhere, otherwise prefer a single implementation. Function names and static methods of classes starts with Capital letter. Local functions names ends with underscore meaning "private". Names of variables must be explicative and tell what the variable does. Short names are not allowed unless "very local in scope". Use underscore for those variables and preferably Tmp within the name. For codes that are intended to be algorithms of some kind (e.g. not plots or things to run on the host PC), make them always MATLAB codegen safe (especially if codegen directive is used). In that case names should be limited to 31 chars. Add the same template of doc to functions as below and always specify arguments-end block for input and output:
%% SIGNATURE
%
% -------------------------------------------------------------------------------------------------------------
%% DESCRIPTION
% -------------------------------------------------------------------------------------------------------------
%% INPUT
% -------------------------------------------------------------------------------------------------------------
%% OUTPUT
% -------------------------------------------------------------------------------------------------------------
%% CHANGELOG
% Nov-Dec 2024  Pietro Califano     First prototype.
% -------------------------------------------------------------------------------------------------------------
%% DEPENDENCIES
%
% -------------------------------------------------------------------------------------------------------------

%% Function code
