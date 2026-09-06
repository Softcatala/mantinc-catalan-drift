TASK ?= catalan_drift
RUN_NAME ?= lm-eval
LM_EVAL_MODEL ?= hf
MODEL_ARGS ?=
GEN_KWARGS ?= {"temperature":0,"reasoning_effort":"none","chat_template_kwargs":{"enable_thinking":false}}
DISPLAY_MODEL ?= $(MODEL_ARGS)
MODEL_OUT_DIR ?= outputs/$(shell printf '%s' '$(RUN_NAME)' | tr '[:upper:]' '[:lower:]')
OUT_DIR ?= $(MODEL_OUT_DIR)/lm_eval
EVAL_TIMELINE ?= outputs/eval_timeline.tsv
PROMPTS ?= data/prompts_monolingual.yaml data/prompts_crosslingual_basic.yaml data/prompts_multi_turn.yaml data/prompts_crosslingual_advanced.yaml data/prompts_rag_context.yaml
EXPORT ?= data/lm_eval/catalan_drift.jsonl
EVAL_RUNS ?= gpt-5.6 Ministral-3-8B-Instruct-2512-Q4_K_M
CLOUD_EVAL_TARGETS ?= eval-gpt56
LOCAL_EVAL_TARGETS ?= eval-ministral3-8b
AI_LOCAL_MODELS ?= \
	Muse-Glimmer-30B-UD-Q4_K_XL \
	google_gemma-4-26B-A4B-it-Q4_K_M \
	google_gemma-3-27b-it-Q4_K_M \
	google_gemma-3-12b-it-Q4_K_M \
	mistralai_Mistral-Small-3.2-24B-Instruct-2506-Q4_K_M \
	Meta-Llama-3.1-8B-Instruct-Q4_K_M \
	google_gemma-4-E4B-it-Q4_K_M \
	phi-4-Q4_K_M \
	EuroLLM-9B-Instruct-Q4_K_M \
	google_gemma-3-4b-it-Q4_K_M \
	aya-expanse-8b-Q4_K_M \
	Ministral-3-8B-Instruct-2512-Q4_K_M \
	ALIA-7b-fc-2607-Q4_0
LOCAL_OPENAI_BASE_URL ?= http://localhost:9090/v1/chat/completions
LOCAL_NUM_CONCURRENT ?= 1
GPT_GEN_KWARGS ?= {"temperature":0,"reasoning_effort":"none"}
GEMMA_NO_THINKING_GEN_KWARGS ?= {"temperature":0,"reasoning_effort":"none","chat_template_kwargs":{"enable_thinking":false}}
LOCAL_GEN_KWARGS ?= {"temperature":0,"reasoning_effort":"none","chat_template_kwargs":{"enable_thinking":false}}
UV_CACHE_DIR ?= .uv-cache
UV_PYTHON_INSTALL_DIR ?= .uv-python
UV_EXTRAS ?= api
UV_EXTRA_FLAGS := $(foreach extra,$(UV_EXTRAS),--extra $(extra))
UV_RUN ?= UV_CACHE_DIR=$(UV_CACHE_DIR) UV_PYTHON_INSTALL_DIR=$(UV_PYTHON_INSTALL_DIR) uv run $(UV_EXTRA_FLAGS)
PYTHON ?= $(UV_RUN) python
LM_EVAL ?= $(UV_RUN) lm_eval
LANGUAGE_ID_MODEL ?= models/lid.176.ftz
LANGUAGE_ID_MODEL_URL ?= https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.ftz
FLORES_DIR ?= data/flores200
FLORES_ARCHIVE ?= $(FLORES_DIR)/flores200_dataset.tar.gz
FLORES_URL ?= https://dl.fbaipublicfiles.com/nllb/flores200_dataset.tar.gz
FLORES_LANGS ?= cat_Latn spa_Latn eng_Latn
SKIP_EXPORT ?=
LIMIT ?=
EVAL_EXPORT_PREREQ := $(if $(SKIP_EXPORT),,export-lm-eval)
HF_DATASET_REPO ?= softcatala/mantinc-catalan-drift
HF_DATASET_METADATA ?= data/lm_eval/catalan_drift.metadata.yaml
HF_DATASET_FILES ?= data/lm_eval/catalan_drift.jsonl $(HF_DATASET_METADATA)
VERSION ?=

ifneq (,$(filter publish-dataset,$(MAKECMDGOALS)))
ifeq (,$(VERSION))
$(error Set VERSION=X.Y.Z (e.g. VERSION=1.1.0))
endif
ifeq (,$(shell printf '%s' '$(VERSION)' | grep -E '^[0-9]+\.[0-9]+\.[0-9]+$$'))
$(error VERSION must match X.Y.Z format (e.g. 1.1.0); got: $(VERSION))
endif
endif

.PHONY: build test clean-outputs language-id-model flores-corpus export-lm-eval eval eval-one eval-local-openai eval-summary all_ai_local_models publish-dataset
.PHONY: $(CLOUD_EVAL_TARGETS) $(LOCAL_EVAL_TARGETS)

build:
	$(PYTHON) scripts/build_dataset.py

test:
	$(UV_RUN) pytest -q

language-id-model: $(LANGUAGE_ID_MODEL)
	@echo "Language ID model ready: $(LANGUAGE_ID_MODEL)"

$(LANGUAGE_ID_MODEL):
	@mkdir -p "$$(dirname "$@")"
	curl -L "$(LANGUAGE_ID_MODEL_URL)" -o "$@.tmp"
	mv "$@.tmp" "$@"

flores-corpus: $(FLORES_DIR)/devtest/cat_Latn.devtest
	@echo "FLORES-200 corpus ready under $(FLORES_DIR)"

$(FLORES_DIR)/devtest/cat_Latn.devtest: $(FLORES_ARCHIVE)
	@mkdir -p "$(FLORES_DIR)"
	tar xzf "$(FLORES_ARCHIVE)" -C "$(FLORES_DIR)" --transform='s|^\./flores200_dataset/||' \
		$(foreach lang,$(FLORES_LANGS),./flores200_dataset/dev/$(lang).dev ./flores200_dataset/devtest/$(lang).devtest)

$(FLORES_ARCHIVE):
	@mkdir -p "$(FLORES_DIR)"
	curl -L "$(FLORES_URL)" -o "$@.tmp"
	mv "$@.tmp" "$@"

clean-outputs:
	@echo "Clearing outputs/"
	@mkdir -p outputs
	@find outputs -mindepth 1 -maxdepth 1 -exec rm -rf {} +

export-lm-eval: build
	$(PYTHON) scripts/catalan_drift_eval.py export-lm-eval --prompts $(PROMPTS) --output "$(EXPORT)"

eval: clean-outputs language-id-model export-lm-eval
	$(MAKE) -j2 SKIP_EXPORT=1 $(CLOUD_EVAL_TARGETS) & \
	$(MAKE) SKIP_EXPORT=1 $(LOCAL_EVAL_TARGETS) & \
	wait
	$(MAKE) eval-summary

eval-summary:
	$(PYTHON) scripts/catalan_drift_eval.py summary-lm-eval --task "$(TASK)" --timeline "$(EVAL_TIMELINE)" --runs $(EVAL_RUNS)

eval-one:
	@test -n "$(MODEL_ARGS)" || (echo "Set MODEL_ARGS" && exit 2)
	@mkdir -p "$$(dirname "$(EVAL_TIMELINE)")"
	@start=$$(date +%s); \
	start_iso=$$(date '+%Y-%m-%dT%H:%M:%S%z'); \
	git_commit=$$(git rev-parse HEAD); \
	printf '[%s] eval start: %s\n' "$(DISPLAY_MODEL)" "$$start_iso"; \
	printf '%s\t%s\t%s\t%s\t%s\n' "$(RUN_NAME)" "$(DISPLAY_MODEL)" start "$$start_iso" "" >> "$(EVAL_TIMELINE)"; \
	if $(LM_EVAL) --include_path lm_eval_tasks --tasks "$(TASK)" --model "$(LM_EVAL_MODEL)" --model_args "$(MODEL_ARGS)" --apply_chat_template --log_samples --output_path "$(OUT_DIR)" $(if $(LIMIT),--limit "$(LIMIT)",) $(if $(GEN_KWARGS),--gen_kwargs '$(GEN_KWARGS)',) && samples=$$(find "$(OUT_DIR)" -name 'samples_$(TASK)*.jsonl' | sort | tail -n 1) && $(PYTHON) scripts/catalan_drift_eval.py score-lm-eval --samples "$$samples" --model "$(DISPLAY_MODEL)" --provider "$(LM_EVAL_MODEL)" --responses-output "$(MODEL_OUT_DIR)/responses.jsonl" --report "$(MODEL_OUT_DIR)/report.json" --failures-file "$(MODEL_OUT_DIR)/failures.txt" --passes-file "$(MODEL_OUT_DIR)/passes.txt" --git-commit "$$git_commit"; then status=0; event=end; else status=$$?; event=failed; fi; \
	end_iso=$$(date '+%Y-%m-%dT%H:%M:%S%z'); \
	elapsed=$$(($$(date +%s) - start)); \
	printf '[%s] eval %s: %s (duration %ss)\n' "$(DISPLAY_MODEL)" "$$event" "$$end_iso" "$$elapsed"; \
	printf '%s\t%s\t%s\t%s\t%s\n' "$(RUN_NAME)" "$(DISPLAY_MODEL)" "$$event" "$$end_iso" "$$elapsed" >> "$(EVAL_TIMELINE)"; \
	exit $$status

eval-local-openai: $(EVAL_EXPORT_PREREQ)
	@test -n "$(DISPLAY_MODEL)" || (echo "Set DISPLAY_MODEL, for example: make eval-local-openai DISPLAY_MODEL=gemma-4-12b-it-Q4_K_M" && exit 2)
	OPENAI_API_KEY=local $(MAKE) eval-one LM_EVAL_MODEL=local-chat-completions MODEL_ARGS="model=$(DISPLAY_MODEL),base_url=$(LOCAL_OPENAI_BASE_URL),tokenized_requests=False,num_concurrent=$(LOCAL_NUM_CONCURRENT)" DISPLAY_MODEL="$(DISPLAY_MODEL)" RUN_NAME="$(DISPLAY_MODEL)" GEN_KWARGS='$(GEN_KWARGS)'

all_ai_local_models: $(EVAL_EXPORT_PREREQ)
	@set -e; \
	for model in $(AI_LOCAL_MODELS); do \
		$(MAKE) SKIP_EXPORT=1 eval-local-openai DISPLAY_MODEL="$$model"; \
	done

eval-gpt56: $(EVAL_EXPORT_PREREQ)
	$(MAKE) eval-one LM_EVAL_MODEL=openai-chat-completions MODEL_ARGS="model=gpt-5.6,num_concurrent=4" DISPLAY_MODEL=gpt-5.6 RUN_NAME=gpt-5.6 GEN_KWARGS='$(GPT_GEN_KWARGS)'

eval-ministral3-8b:
	$(MAKE) eval-local-openai DISPLAY_MODEL=Ministral-3-8B-Instruct-2512-Q4_K_M LOCAL_NUM_CONCURRENT=2 GEN_KWARGS='$(LOCAL_GEN_KWARGS)'
eval-qwen3-14b:
	$(MAKE) eval-local-openai DISPLAY_MODEL=Qwen_Qwen3-14B-Q4_K_M LOCAL_NUM_CONCURRENT=2

publish-dataset:
	@test -z "$$(git status --porcelain)" || { \
		echo "Working tree must be clean before preparing a release."; \
		git status --short; \
		exit 2; \
	}
	@git show-ref --verify --quiet "refs/tags/$(VERSION)" && { echo "Git tag already exists: $(VERSION)"; exit 2; } || status=$$?; \
		test "$${status:-0}" -eq 1
	$(MAKE) export-lm-eval
	@checksum=$$(sha256sum "$(EXPORT)" | cut -d ' ' -f 1); \
		printf 'version: "%s"\nfile: "%s"\nchecksum_sha256: "%s"\ncreated: "%s"\n' \
		"$(VERSION)" "$$(basename "$(EXPORT)")" "$$checksum" "$$(date --iso-8601=seconds)" > "$(HF_DATASET_METADATA)"
	@for f in $(HF_DATASET_FILES); do \
		test -f "$$f" || { echo "Missing file: $$f"; exit 2; }; \
	done
	@git status --short -- $(HF_DATASET_FILES)
	@git diff -- $(HF_DATASET_FILES)
	@printf 'Create Git commit "Release $(VERSION)" and continue publishing? [y/N] '; \
		read -r answer; \
		case "$$answer" in y|Y) ;; *) echo "Release cancelled."; exit 2;; esac
	git add -- $(HF_DATASET_FILES)
	git commit -m "Release $(VERSION)"
	@test -z "$$(git status --porcelain)" || { echo "Working tree is not clean after the release commit; stopping."; exit 2; }
	@for f in $(HF_DATASET_FILES); do \
		echo "Uploading $$f to $(HF_DATASET_REPO)"; \
		hf upload --repo-type dataset "$(HF_DATASET_REPO)" "$$f" "$$f" --commit-message "Release $(VERSION)" || exit $$?; \
	done
	@echo "Tagging $(HF_DATASET_REPO) as $(VERSION)"
	$(PYTHON) -c "from huggingface_hub import HfApi; HfApi().create_tag('$(HF_DATASET_REPO)', tag='$(VERSION)', repo_type='dataset')"
	@echo "Tagging Git repository as $(VERSION)"
	git tag "$(VERSION)"
