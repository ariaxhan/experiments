# K:REFACTOR - Refactoring Kernel

●kernel|K:REFACTOR|role:code_improvement|output:refactored_code|mandate:protect_stability|pattern:WF:REFACTOR

## Workflow

→WF:STEP0|analyze:current_structure|identify:improvements|assess:risks|create:refactor_blueprint
→WF:STEP1|review:current_code|quality:assurance|identify:risks|pattern:PT:102|verify:stability
→WF:STEP2|implement:refactor|incremental:changes|protect:stability|verify:after_each|test:regression
→WF:STEP3|extract:refactor_patterns|update:patterns.mdc|delete:blueprint|pattern:PT:69
→WF:FINAL|verify:functionality|test:regression|status:S:COMPLETE

## Mandate

→mandate|protect_stability:critical|incremental:changes|verify:after_each|test:regression
→refactoring|improve:structure|maintain:functionality|protect:stability|never:break_existing
→incremental|small:changes|verify:after_each|test:regression|protect:critical_paths
→verification|functionality:works|regression:test|stability:maintained|never:assume_works

## Refactoring Patterns

●pattern|WF:REFACTOR|refactoring_workflow|context:K:REFACTOR|usage:code_improvement
→step0|analyze:current_structure|identify:improvements|create:refactor_blueprint|assess:risks
→step1|review:current_code|quality:assurance|identify:risks|pattern:PT:102|verify:stability
→step2|implement:refactor|incremental:changes|protect:stability|verify:after_each|test:regression
→step3|extract:refactor_patterns|update:patterns.mdc|delete:blueprint|pattern:PT:69

## Stability Protection

●stability_protection|mandatory:true|critical:true
→before_refactor|test:existing_functionality|verify:current_state|protect:critical_paths|document:baseline
→during_refactor|incremental:changes|verify:after_each|test:regression|isolate:scope
→after_refactor|test:regression|verify:functionality|validate:improvements|protect:stability
→never|break_existing:allowed|assume_works:allowed|skip_tests:allowed|always:verify

## Refactoring Types

●refactoring_types|common:patterns|universal:true
→extract_method|isolate:logic|improve:readability|maintain:functionality|verify:works
→rename_variable|improve:clarity|update:references|verify:all_updated|test:regression
→restructure_class|improve:organization|maintain:interfaces|verify:functionality|test:regression
→optimize_imports|improve:organization|verify:all_work|test:functionality|never:break_imports

○reference|philosophy:philosophy.mdc|global:global.mdc|patterns:patterns_extended.mdc|workflows:workflows.mdc

