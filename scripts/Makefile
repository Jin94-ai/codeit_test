# scripts/Makefile

DATA_DIR := ./data
DATASETS_DIR := ./datasets
RUNS_DIR := ./runs
RUN_SCRIPT := ./scripts/run.sh

clean_hard:
	# data/ 내부 정리 (.md, .gitkeep 제외)
	find $(DATA_DIR) -type f \
		-not -name "*.md" \
		-not -name ".gitkeep" \
		-delete

	find $(DATA_DIR) -type d -empty \
		-not -path "$(DATA_DIR)" \
		-delete

	# datasets/ 전체 삭제
	if [ -d $(DATASETS_DIR) ]; then rm -rf $(DATASETS_DIR); fi

	# runs/ 전체 삭제
	if [ -d $(RUNS_DIR) ]; then rm -rf $(RUNS_DIR); fi

	# scripts/run.sh 삭제
	if [ -f $(RUN_SCRIPT) ]; then rm -f $(RUN_SCRIPT); fi

	# 프로젝트 전체의 .pt 파일 삭제
	find . -type f -name "*.pt" -delete

	@echo "[clean_data] data/, datasets/, runs/, run.sh, *.pt 정리 완료"

clean_soft:
	# scripts/run.sh 삭제
	if [ -f $(RUN_SCRIPT) ]; then rm -f $(RUN_SCRIPT); fi
		# runs/ 전체 삭제
	if [ -d $(RUNS_DIR) ]; then rm -rf $(RUNS_DIR); fi