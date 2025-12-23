# K:DEBUG - Bug Fixing Kernel

●kernel|K:DEBUG|role:bug_fixing|output:fixes|mandate:triage_fix|pattern:PT:22

## Workflow

→WF:STEP0|triage:bug_report|identify:root_cause|reproduce:error|isolate:problem_scope
→WF:STEP1|fix:bug|protect:stability|verify:fix_works|test:regression
→WF:STEP2|verify:imports_types|test:functionality|validate:no_side_effects
→WF:STEP3|extract:fix_pattern|update:patterns.mdc|handoff:to_K:LEARN
→WF:FINAL|cleanup:debug_artifacts|status:S:COMPLETE

## Mandate

→mandate|triage_fix:critical|protect_stability:always|pattern:PT:22|verify:after_fix
→triage|identify:root_cause|reproduce:error|isolate:scope|prioritize:critical_paths
→fix|minimal:changes|targeted:solution|protect:stability|verify:no_side_effects
→verification|test:fix_works|test:regression|verify:imports_types|validate:functionality

## Pattern:PT:22 - Debug Triage Fix

●pattern|PT:22|debug:triage_fix|mandatory:true
→triage|identify:root_cause|reproduce:error|isolate:problem_scope|prioritize:critical
→fix|minimal:changes|targeted:solution|protect:stability|verify:works
→extract|fix_pattern:to_patterns.mdc|root_cause:to_patterns.mdc|solution:to_patterns.mdc

## Debug Protocol

●debug|protocol:systematic|mandatory:true
→reproduce|error:exact_steps|environment:verify|dependencies:check
→isolate|scope:minimal|test:incremental|verify:each_step
→fix|minimal:changes|targeted:solution|protect:stability|never:break_existing
→verify|test:fix_works|test:regression|verify:imports_types|validate:functionality
→extract|pattern:to_patterns.mdc|root_cause:to_patterns.mdc|solution:to_patterns.mdc

## Stability Protection

●stability|protection:mandatory|never:break_existing
→before_fix|test:existing_functionality|verify:current_state|protect:critical_paths
→during_fix|minimal:changes|targeted:solution|isolate:scope|never:refactor_unrelated
→after_fix|test:regression|verify:no_side_effects|validate:functionality|protect:stability

○reference|philosophy:philosophy.mdc|global:global.mdc|patterns:patterns.mdc|symbols:a2a_symbol_dictionary.mdc

