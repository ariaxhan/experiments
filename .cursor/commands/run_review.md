# K:AUDIT - Verification Kernel

●kernel|K:AUDIT|role:verification|output:reviews|mandate:quality_assurance|pattern:PT:102

## Workflow

→WF:STEP0|analyze:code_quality|check:patterns_compliance|verify:type_safety|audit:architecture
→WF:STEP1|review:constructive_destructive|identify:anti_patterns|verify:best_practices
→WF:STEP2|generate:review_report|format:vector_native_lt_20_lines|consolidate:insights
→WF:STEP3|extract:quality_insights|update:patterns.mdc|handoff:to_K:LEARN
→WF:FINAL|cleanup:review_artifacts|status:S:COMPLETE

## Mandate

→mandate|quality_assurance:critical|pattern:PT:102|format:vector_native_lt_20_lines
→review|code_quality:comprehensive|patterns:compliance|type_safety:verification|architecture:audit
→output|review_report:vector_native_lt_20_lines|consolidate:insights|extract:to_patterns.mdc
→extract|quality_insights:to_patterns.mdc|anti_patterns:to_patterns.mdc|best_practices:to_patterns.mdc

## Pattern:PT:102 - Audit Quality Assurance

●pattern|PT:102|audit:quality_assurance|mandatory:true
→analyze|code_quality:comprehensive|patterns:compliance|type_safety:verification|architecture:audit
→review|constructive_destructive:patterns|anti_patterns:identify|best_practices:verify
→extract|quality_insights:to_patterns.mdc|anti_patterns:to_patterns.mdc|best_practices:to_patterns.mdc

## Review Checklist

●review|checklist:comprehensive|mandatory:true
→code_quality|type_hints:mandatory|docstrings:comprehensive|error_handling:proper|modular:architecture
→patterns|constructive:follow|destructive:never|reference:patterns.mdc|compliance:verify
→type_safety|annotations:mandatory|dataclasses:preferred|validation:post_init|never:untyped_public_api
→architecture|separation:concerns|modular:design|interfaces:clear|contracts:defined
→device_handling|automatic:get_device|fallback:mps_cuda_cpu|memory:cleanup|context:managers

## Review Output Format

●review_output|format:vector_native_lt_20_lines|mandatory:true
→summary|quality_score:rating|critical_issues:count|recommendations:priority
→findings|constructive:patterns_found|destructive:anti_patterns_found|improvements:suggested
→extraction|to_patterns.mdc:insights|to_philosophy.mdc:principles|to_global.mdc:architecture

○reference|philosophy:philosophy.mdc|global:global.mdc|patterns:patterns.mdc|symbols:a2a_symbol_dictionary.mdc

