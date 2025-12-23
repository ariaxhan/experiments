# Quick Reference - Kernel Commands & Patterns

●quick_reference|kernel_commands:all|patterns:common|universal:true

## Kernel Commands

●commands|invoke:via_@command|universal:true
→@run_planner|kernel:K:ARCH|role:strategic_planning|output:blueprints|pattern:PT:1_PT:101_PT:103
→@run_executor|kernel:K:EXEC|role:implementation|output:code|pattern:PT:51_PT:52_PT:53
→@run_document_and_learn|kernel:K:LEARN|role:system_evolution|output:updated_rules|pattern:PT:69|auto:enabled
→@run_debugger|kernel:K:DEBUG|role:bug_fixing|output:fixes|pattern:PT:22_PT:151_PT:152
→@run_review|kernel:K:AUDIT|role:verification|output:reviews|pattern:PT:102
→@run_test|kernel:K:TEST|role:testing_verification|output:tests|pattern:PT:251_PT:252
→@run_refactor|kernel:K:REFACTOR|role:code_improvement|output:refactored_code|pattern:WF:REFACTOR
→@run_handoff|purpose:lossless_continuity|format:vector_native_markdown|location:plans/[username]/active/handoffs/

## Common Workflow Patterns

●workflows|common:patterns|universal:true
→WF:NEW_FEATURE|K:ARCH→K:EXEC→K:LEARN|plan:implement:extract|standard:workflow
→WF:BUG_FIX|K:DEBUG→K:LEARN|fix:extract_pattern|pattern:PT:22_then_PT:69|quick:workflow
→WF:REFACTOR|K:ARCH→K:AUDIT→K:EXEC→K:LEARN|plan:review:implement:extract|protect:stability
→WF:CODE_REVIEW|K:AUDIT→K:LEARN|review:extract_insights|pattern:PT:102_then_PT:69|quality:assurance

## Core Patterns (Quick Lookup)

●patterns|core:PT:1_to_PT:102|universal:true
→PT:1|investigate_first|context:K:ARCH|usage:always_before_planning|mandate:verify_first
→PT:22|debug_triage_fix|context:K:DEBUG|usage:bug_fixing|triage:fix:extract
→PT:51|pre_implementation_verification|context:K:EXEC|usage:before_code|check:imports_types_dependencies
→PT:52|post_implementation_verification|context:K:EXEC|usage:after_each_file|verify:imports_types_functionality
→PT:53|stability_protection|context:K:EXEC_K:DEBUG|usage:always|protect:critical_paths
→PT:69|consolidation_extract_delete|context:K:LEARN|usage:after_each_phase|extract:delete:update
→PT:101|blueprint_structure|context:K:ARCH|usage:blueprint_creation|format:vector_native_lt_30_lines
→PT:102|audit_quality_assurance|context:K:AUDIT|usage:verification|analyze:review:extract
→PT:103|blueprint_lifecycle|context:K:ARCH→K:EXEC→K:LEARN|usage:always|create:execute:extract:delete

## Status Markers

●status|markers:standard|universal:true
→S:ACTIVE|state:in_progress|context:current_work|tracking:blueprints_handoffs
→S:READY|state:awaiting_execution|context:blueprints_complete|next:K:EXEC
→S:COMPLETE|state:finished|context:extracted_deleted|next:K:LEARN

## Vector Native Syntax

●syntax|format:vector_native|mandatory:true
→●|directive:core_concept|usage:principles_rules_definitions
→→|action:sub_directive|usage:specifications_requirements
→○|reference:external_concept|usage:cross_references_patterns
→⊕|addition:enhancement|usage:extensions_modifications

## Common Kernel Combinations

●combinations|standard:patterns|universal:true
→K:ARCH→K:EXEC|planning:then_implementation|blueprint:contract|handoff:S:READY
→K:EXEC→K:LEARN|implementation:then_extraction|auto:enabled|pattern:PT:69
→K:DEBUG→K:LEARN|fix:then_extract_pattern|triage:PT:22|extract:PT:69
→K:AUDIT→K:LEARN|review:then_extract_insights|quality:PT:102|extract:PT:69
→K:ARCH→K:AUDIT→K:EXEC|plan:review:implement|verify:before_execution|protect:stability

## Error Recovery

●error_recovery|patterns:standard|universal:true
→verification_failure|rollback:changes|investigate:root_cause|triage:K:DEBUG|pattern:PT:22
→import_error|check:dependencies|verify:versions|update:requirements|never:assume_available
→type_error|verify:annotations|check:dataclasses|validate:post_init|never:untyped_public_api
→stability_break|rollback:immediately|investigate:root_cause|fix:minimal|protect:critical_paths

○reference|philosophy:philosophy.mdc|global:global.mdc|symbols:a2a_symbol_dictionary.mdc|patterns:patterns_extended.mdc|workflows:workflows.mdc|integration:integration.mdc

