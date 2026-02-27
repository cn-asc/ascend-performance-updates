#!/usr/bin/env bash
set -euo pipefail

# ─────── VARIABLES ─────────────────────────────────────────────────────────────
PROJECT=${PROJECT:-$(gcloud config get-value project)}
EXPECTED_PROJECT="investmentprocessor"
TZ="America/New_York"

# Verify we're deploying to the correct project
if [ "$PROJECT" != "$EXPECTED_PROJECT" ]; then
  echo "⚠️  Warning: Expected project '${EXPECTED_PROJECT}', but current project is '${PROJECT}'"
  read -p "Continue anyway? (y/N) " -n 1 -r
  echo
  if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    exit 1
  fi
fi

##### Investment Updates Pipeline
JOB_NAME="investment-updates-pipeline"
REGION="us-central1"
SERVICE_ACCOUNT="${JOB_NAME}-sa@${PROJECT}.iam.gserviceaccount.com"
SCHEDULE="0 9 * * *"    # Once daily at 9:00 AM EST
ENV_FILE="env.yaml"

# ─────── 1) Create service account if it doesn't exist ────────────────────────
echo "⏳ Checking service account..."
if ! gcloud iam service-accounts describe "${SERVICE_ACCOUNT}" --project="${PROJECT}" &>/dev/null; then
  echo "⏳ Creating service account..."
  gcloud iam service-accounts create "${JOB_NAME}-sa" \
    --project="${PROJECT}" \
    --display-name="Investment Updates Pipeline Service Account" \
    --description="Service account for investment updates automation pipeline"
  
  # Grant necessary permissions
  echo "⏳ Granting permissions..."
  gcloud projects add-iam-policy-binding "${PROJECT}" \
    --member="serviceAccount:${SERVICE_ACCOUNT}" \
    --role="roles/run.invoker" --quiet
  
  gcloud projects add-iam-policy-binding "${PROJECT}" \
    --member="serviceAccount:${SERVICE_ACCOUNT}" \
    --role="roles/logging.logWriter" --quiet
  
  echo ""
  echo "⚠️  IMPORTANT: For Google Drive access, you need to:"
  echo "   1. Share the Drive folders/files with: ${SERVICE_ACCOUNT}"
  echo "   2. OR set up domain-wide delegation (for Workspace domains)"
  echo ""
fi

# ─────── 2) Deploy Cloud Run Job ──────────────────────────────────────────────
echo "⏳ Deploying Cloud Run Job ${JOB_NAME}..."
gcloud run jobs deploy "${JOB_NAME}" \
  --project="${PROJECT}" \
  --region="${REGION}" \
  --service-account="${SERVICE_ACCOUNT}" \
  --source=. \
  --env-vars-file="${ENV_FILE}" \
  --command="python3" \
  --args="main.py" \
  --memory=2Gi \
  --cpu=2 \
  --max-retries=1 \
  --task-timeout=900 \
  --quiet

# ─────── 3) Create Cloud Scheduler Job ────────────────────────────────────────
echo "⏳ Configuring Cloud Scheduler job..."
if gcloud scheduler jobs describe "${JOB_NAME}-scheduler" \
  --project="${PROJECT}" \
  --location="${REGION}" &>/dev/null; then
  ACTION="update"
else
  ACTION="create"
fi

COMMON_FLAGS=(
  --project="${PROJECT}"
  --location="${REGION}"
  --schedule="${SCHEDULE}"
  --time-zone="${TZ}"
  --http-method=POST
  --uri="https://${REGION}-run.googleapis.com/apis/run.googleapis.com/v1/namespaces/${PROJECT}/jobs/${JOB_NAME}:run"
  --oidc-service-account-email="${SERVICE_ACCOUNT}"
  --oidc-token-audience="https://${REGION}-run.googleapis.com/apis/run.googleapis.com/v1/namespaces/${PROJECT}/jobs/${JOB_NAME}:run"
)

if [[ "$ACTION" == "update" ]]; then
  gcloud scheduler jobs update http "${JOB_NAME}-scheduler" "${COMMON_FLAGS[@]}" --quiet
  gcloud scheduler jobs resume "${JOB_NAME}-scheduler" \
    --project="${PROJECT}" \
    --location="${REGION}" \
    --quiet
else
  gcloud scheduler jobs create http "${JOB_NAME}-scheduler" "${COMMON_FLAGS[@]}" --quiet
fi

# ─────── 4) Summary ──────────────────────────────────────────────────────────────
echo ""
echo "✅ Cloud Run Job '${JOB_NAME}' deployed"
echo "✅ Cloud Scheduler '${JOB_NAME}-scheduler' → ${SCHEDULE} (${TZ})"
echo ""
echo "To manually run the job:"
echo "  gcloud run jobs execute ${JOB_NAME} --region=${REGION} --project=${PROJECT}"
echo ""
echo "To view logs:"
echo "  gcloud logging read \"resource.type=cloud_run_job AND resource.labels.job_name=${JOB_NAME}\" --limit=50 --project=${PROJECT}"
