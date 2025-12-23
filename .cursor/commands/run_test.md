# K:TEST - Testing Kernel

●kernel|K:TEST|role:testing_verification|output:tests|mandate:comprehensive_coverage|pattern:PT:251_PT:252

## Workflow

→WF:STEP0|analyze:code_structure|identify:test_targets|design:test_strategy|coverage:plan
→WF:STEP1|create:test_files|structure:test_[module].py|classes:Test[Class]|methods:test_[function]
→WF:STEP2|implement:tests|happy_path:test|edge_cases:test|error_cases:test|coverage:comprehensive
→WF:STEP3|run:tests|verify:all_pass|coverage:check|extract:test_patterns|pattern:PT:69
→WF:FINAL|update:patterns.mdc|document:test_patterns|status:S:COMPLETE

## Mandate

→mandate|comprehensive_coverage:critical|pattern:PT:251_PT:252|verify:all_pass
→coverage|happy_path:mandatory|edge_cases:mandatory|error_cases:mandatory|never:untested_code
→structure|test_file:test_[module].py|test_class:Test[Class]|test_methods:test_[function]
→verification|all_tests:pass|coverage:acceptable|functionality:works|error_handling:proper

## Test Patterns

●pattern|PT:251|unit_test_pattern|context:K:TEST|usage:implementation
→structure|test_file:test_[module].py|test_class:Test[Class]|test_methods:test_[function]
→coverage|happy_path:test|edge_cases:test|error_cases:test|never:untested_code
→verify|imports:work|types:correct|functionality:works|error_handling:proper

●pattern|PT:252|integration_test_pattern|context:K:TEST|usage:integration
→test|module_integration:works|data_flow:correct|error_propagation:proper|stability:maintained
→verify|end_to_end:works|error_handling:comprehensive|performance:acceptable|protect:stability

## Test Structure

●test_structure|mandatory:true|universal:true
→file_naming|test_[module].py|convention:standard|location:tests/|imports:proper
→class_structure|Test[Class]|setup:if_needed|teardown:if_needed|methods:test_[function]
→method_naming|test_[function]_[scenario]|descriptive:names|isolated:tests|independent:tests

## Coverage Requirements

●coverage|requirements:mandatory|universal:true
→happy_path|test:normal_usage|verify:expected_output|verify:no_errors|mandatory:true
→edge_cases|test:boundary_conditions|test:empty_inputs|test:max_inputs|mandatory:true
→error_cases|test:invalid_inputs|test:missing_dependencies|test:error_handling|mandatory:true
→never|untested_code:allowed|skip_tests:allowed|assume_works:allowed|always:verify

○reference|philosophy:philosophy.mdc|global:global.mdc|patterns:patterns_extended.mdc|workflows:workflows.mdc

