isort:
	isort

black:
	black .

quality-check: isort black

build:
	docker compose build
	@echo "Waiting for 15 seconds..."
	@sleep 15

init: build
	docker compose up airflow-init

run:
	docker compose up -d

stop:
	docker compose down

