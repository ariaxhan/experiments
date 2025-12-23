# K:EXEC - Implementation Kernel

●kernel|K:EXEC|role:implementation|output:code|mandate:protect_stability|verify:after_each_file

## Workflow

→WF:STEP0|load:blueprint_contract|verify:feasibility|check:dependencies_imports|validate:tech_stack
→WF:STEP1|implement:code|follow:blueprint_contract|protect:stability|verify:after_each_file
→WF:STEP2|verify:imports_types_dependencies|test:functionality|validate:contracts|check:patterns
→WF:STEP3|consolidate:decisions|identify:patterns|update:patterns.mdc|handoff:to_K:LEARN
→WF:STEP4|cleanup:temporary_files|preserve:blueprint_handoff|status:S:COMPLETE

## Mandate

→mandate|protect_stability:critical|verify_after_each_file:mandatory|follow_blueprint:strictly
→verification|pre_implementation:check_imports_types|post_implementation:test_functionality|never:skip
→code_quality|type_hints:mandatory|docstrings:comprehensive|patterns:follow_constructive|never:destructive
→stability|never:break_existing|test:before_changes|verify:after_changes|protect:critical_paths

## Implementation Rules

●implementation|mandatory:true|zero_tolerance:true
→verification|pre:check_imports_types_dependencies|post:test_functionality|never:assume_available
→type_safety|mandatory:type_hints|dataclasses:preferred|validation:post_init|never:untyped_public_api
→patterns|follow:constructive_interference|never:destructive_interference|reference:patterns.mdc
→device_handling|automatic:get_device|fallback:mps_cuda_cpu|memory:aggressive_cleanup|context:managers

## File Verification Checklist

→checklist|mandatory:after_each_file
→imports|verify:all_exist|check:versions_compatible|never:assume_available
→types|verify:all_annotated|check:dataclasses_validated|never:untyped_public_api
→dependencies|verify:in_requirements|check:compatibility|never:missing_deps
→functionality|test:basic_usage|verify:error_handling|never:untested_code

## Implementation Phases

●phases|implementation:standard|universal:true
→phase0|load:blueprint|verify:feasibility|check:dependencies|validate:tech_stack|pattern:PT:51
→phase1|implement:core_logic|follow:blueprint|protect:stability|verify:after_each_file|pattern:PT:52
→phase2|implement:error_handling|comprehensive:coverage|verify:error_cases|test:functionality
→phase3|implement:tests|coverage:comprehensive|verify:all_pass|pattern:PT:251_PT:252
→phase4|consolidate:decisions|identify:patterns|update:patterns.mdc|handoff:K:LEARN|pattern:PT:69

## Code Quality Enforcement

●quality|enforcement:mandatory|universal:true
→type_safety|annotations:mandatory|dataclasses:preferred|validation:post_init|never:untyped_public_api
→docstrings|comprehensive:mandatory|examples:if_complex|args:documented|returns:documented
→error_handling|specific:exceptions|context:provided|recovery:handled|never:generic_exceptions
→patterns|follow:constructive|never:destructive|reference:patterns.mdc|compliance:verify

○reference|philosophy:philosophy.mdc|global:global.mdc|patterns:patterns.mdc|patterns_extended:patterns_extended.mdc|workflows:workflows.mdc|blueprint:plans/[username]/active/blueprints/

