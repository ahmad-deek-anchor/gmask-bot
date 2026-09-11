#!/usr/bin/env bash
# Deploy (create or update) the daily `signals` snapshot as a Cloud Run job + its Cloud
# Scheduler trigger. Idempotent: re-run after any code change.
#
#   deploy/deploy_snapshot_job.sh            # build from source, deploy job, upsert scheduler
#   SKIP_SCHEDULER=1 deploy/deploy_snapshot_job.sh
#
# What it sets up (project anchorage-corp-eng-playground, region us-east1):
#   * Cloud Run job `trading-signals-snapshot`: the repo's Dockerfile built by Cloud Build
#     (--source .), running `python snapshot_daily.py --sources signals --verbose` as the
#     gm-bot service account with SNAPSHOT_BACKEND=bigquery, so rows land in
#     anchorage-corp-eng-playground.gmask_bot.snapshots. haruko / sheet stay on the local
#     systemd timer: the SA cannot read anc-global-markets and cannot be shared the Sheet.
#   * Cloud Scheduler job `trading-signals-snapshot-daily`: 23:30 UTC daily, POST to the Cloud
#     Run Admin API `jobs.run` endpoint authenticated as the same SA. The Admin API is a Google
#     API, so the token is an OAuth access token (--oauth-service-account-email); an OIDC
#     identity token is only accepted by Cloud Run *services*, not by run.googleapis.com.
#
# IAM the SA already holds (granted out of band): secretmanager.secretAccessor on the
# amberdata_key / coinmetrics_trial_api secrets (anchorage-trading-solutions),
# bigquery.jobUser on the project, bigquery.dataEditor on dataset gmask_bot,
# aiplatform.user on anchorage-ai-development, run.invoker on the project (lets the
# scheduler call jobs.run).
set -euo pipefail

PROJECT="${PROJECT:-anchorage-corp-eng-playground}"
REGION="${REGION:-us-east1}"
JOB="${JOB:-trading-signals-snapshot}"
SCHEDULER_JOB="${SCHEDULER_JOB:-${JOB}-daily}"
SA="${SA:-gm-bot@${PROJECT}.iam.gserviceaccount.com}"
SCHEDULE="${SCHEDULE:-30 23 * * *}"        # UTC; the signals source uses the last complete UTC day
SOURCES="${SOURCES:-signals}"

ENV_VARS="SNAPSHOT_BACKEND=bigquery"
ENV_VARS+=",SNAPSHOT_BQ_TABLE=${SNAPSHOT_BQ_TABLE:-anchorage-corp-eng-playground.gmask_bot.snapshots}"
ENV_VARS+=",GCP_SECRETS_PROJECT=anchorage-trading-solutions"
ENV_VARS+=",BQ_BILLING_PROJECT=${PROJECT}"
ENV_VARS+=",VERTEX_PROJECT=anchorage-ai-development"
ENV_VARS+=",MEMORY_EMBEDDINGS=off"          # the job never writes memories; skip the Vertex probe
ENV_VARS+=",FORCE_IPV4=0"                   # the dev-box IPv6 workaround is not needed on Cloud Run

cd "$(dirname "$0")/.."

echo ">> Deploying Cloud Run job ${JOB} (${PROJECT}/${REGION}) from source ..."
gcloud run jobs deploy "${JOB}" \
  --source . \
  --project "${PROJECT}" --region "${REGION}" \
  --service-account "${SA}" \
  --command python \
  --args "snapshot_daily.py,--sources,${SOURCES},--verbose" \
  --set-env-vars "${ENV_VARS}" \
  --task-timeout 20m \
  --max-retries 1 \
  --memory 1Gi --cpu 1 \
  --tasks 1 \
  --quiet

RUN_URI="https://run.googleapis.com/v2/projects/${PROJECT}/locations/${REGION}/jobs/${JOB}:run"

if [[ "${SKIP_SCHEDULER:-0}" == "1" ]]; then
  echo ">> SKIP_SCHEDULER=1: leaving Cloud Scheduler untouched."
  exit 0
fi

if gcloud scheduler jobs describe "${SCHEDULER_JOB}" --location "${REGION}" --project "${PROJECT}" >/dev/null 2>&1; then
  VERB=update
else
  VERB=create
fi
echo ">> ${VERB^} Cloud Scheduler job ${SCHEDULER_JOB} (${SCHEDULE} UTC -> ${RUN_URI}) ..."
gcloud scheduler jobs "${VERB}" http "${SCHEDULER_JOB}" \
  --location "${REGION}" --project "${PROJECT}" \
  --schedule "${SCHEDULE}" --time-zone "Etc/UTC" \
  --uri "${RUN_URI}" --http-method POST \
  --oauth-service-account-email "${SA}" \
  --oauth-token-scope "https://www.googleapis.com/auth/cloud-platform" \
  --attempt-deadline 30m \
  --description "Daily trading-signals snapshot (signals source) -> BigQuery gmask_bot.snapshots" \
  --quiet

echo ">> Done. Inspect with:"
echo "   gcloud run jobs executions list --job ${JOB} --region ${REGION} --project ${PROJECT}"
echo "   gcloud scheduler jobs describe ${SCHEDULER_JOB} --location ${REGION} --project ${PROJECT}"
