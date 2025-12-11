# Capital-One-Team-Two Fraud Detection System

This repository contains the code and configuration for the **Capital‑One‑Team‑Two—Fraud‑Detection** capstone project for UW‑Madison CS620.  It implements a serverless fraud‑detection pipeline on AWS, a machine‑learning model for predicting fraudulent transactions, and a simple admin dashboard for monitoring accounts and transactions.

## Repository

This will be a public project hosted on GitHub and will be forever available here for reference:  **`https://github.com/Capital-One-Team-Two/Capital-One-Team-Two---Fraud-Detection`** .

## Setting Up the System

The project contains both backend services (Python + AWS) and a frontend dashboard (React).  You will need **node/npm** for the dashboard and **Python 3.8+** (We used Python 3.10+) with AWS credentials for the backend.

1. **Clone the repo and install prerequisites**

   <pre class="overflow-visible! px-0!" data-start="797" data-end="1205"><div class="contain-inline-size rounded-2xl corner-superellipse/1.1 relative bg-token-sidebar-surface-primary"><div class="sticky top-9"><div class="absolute end-0 bottom-0 flex h-9 items-center pe-2"><div class="bg-token-bg-elevated-secondary text-token-text-secondary flex items-center gap-4 rounded-sm px-2 font-sans text-xs"></div></div></div><div class="overflow-y-auto p-4" dir="ltr"><code class="whitespace-pre! language-bash"><span><span>git </span><span>clone</span><span> https://github.com/Capital-One-Team-Two/Capital-One-Team-Two---Fraud-Detection.git
   </span><span>cd</span><span> Capital-One-Team-Two---Fraud-Detection

   </span><span># Python dependencies</span><span>
   python -m venv venv && </span><span>source</span><span> venv/bin/activate
   pip install --upgrade pip
   pip install -r requirements.txt

   </span><span># Front‑end dependencies (admin dashboard)</span><span>
   </span><span>cd</span><span> lambdas/admin/fraud-admin-dashboard
   npm install
   </span></span></code></div></div></pre>
2. **Configure AWS credentials**
   Many scripts and tests require access to AWS services (DynamoDB, Lambda, Step Functions, S3, etc.).  Set `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY` and `AWS_REGION` in your shell or via the AWS CLI.  You will also need to create two DynamoDB tables named `FraudDetection-Transactions` and `FraudDetection-Accounts`.
3. **Deploy the infrastructure**
   Infrastructure as code is located in the `infrastructure/` directory.  It uses the AWS Serverless Application Model (SAM) to define DynamoDB tables, Lambda functions and API Gateway endpoints.

   <pre class="overflow-visible! px-0!" data-start="1797" data-end="1938"><div class="contain-inline-size rounded-2xl corner-superellipse/1.1 relative bg-token-sidebar-surface-primary"><div class="sticky top-9"><div class="absolute end-0 bottom-0 flex h-9 items-center pe-2"><div class="bg-token-bg-elevated-secondary text-token-text-secondary flex items-center gap-4 rounded-sm px-2 font-sans text-xs"></div></div></div><div class="overflow-y-auto p-4" dir="ltr"><code class="whitespace-pre! language-bash"><span><span>cd</span><span> infrastructure
   sam build
   sam deploy --guided
   </span><span># Follow the prompts to choose a stack name, region and parameters</span><span>
   </span></span></code></div></div></pre>

   This creates API endpoints such as `POST /transaction`, `GET /transaction/{id}`, `GET /admin/accounts` and ties them to Lambda functions defined in the `lambdas` folder.
4. **Run the admin dashboard locally**
   The dashboard lives in `lambdas/admin/fraud-admin-dashboard`, implemented with Create React App and Tailwind CSS.  It reads the API Gateway URLs from environment variables in `.env`.  To run locally:

   <pre class="overflow-visible! px-0!" data-start="2361" data-end="2513"><div class="contain-inline-size rounded-2xl corner-superellipse/1.1 relative bg-token-sidebar-surface-primary"><div class="sticky top-9"><div class="absolute end-0 bottom-0 flex h-9 items-center pe-2"><div class="bg-token-bg-elevated-secondary text-token-text-secondary flex items-center gap-4 rounded-sm px-2 font-sans text-xs"></div></div></div><div class="overflow-y-auto p-4" dir="ltr"><code class="whitespace-pre! language-bash"><span><span>cd</span><span> lambdas/admin/fraud-admin-dashboard
   </span><span># edit .env to set REACT_APP_API_BASE_URL to your deployed API Gateway base URL</span><span>
   npm start
   </span></span></code></div></div></pre>

   The application will be available at `http://localhost:3000`.  Enter the internal admin password defined in `src/LoginPage.js` to access the dashboard.  (For demonstration, the password is hard‑coded; in production use a secure & encrypted authentication method.)
5. **Detailed setup guide**
   For additional instructions from setting up the local .env, additional AWS setup details, or even a guide on testing out the functionalities of the system, please refer to the [SETUP_GUIDE.md](SETUP_GUIDE.md) file.

## Overview of the Codebase

* **`infrastructure/`** – CloudFormation/SAM templates describing AWS resources.  Two DynamoDB tables (`FraudDetection-Transactions` and `FraudDetection-Accounts`) store transactions and account metadata.  The template wires up Lambda functions to API Gateway paths and sets up IAM permissions.  Test scripts (e.g., `test_end_to_end_pipeline.py`) show how to invoke the pipeline with boto3.
* **`lambdas/`** – Python code for AWS Lambda.
  * **`transaction/transaction.py`** – Entry point for the `POST /transaction` API.  It writes a new transaction to DynamoDB and triggers fraud scoring.  It uses a LightGBM model loaded from `/opt/ml/model` to compute a fraud probability and stores the result.
  * **`score/score.py`** – Contains the model inference logic and helper functions for converting transactions into features.  The `train_model.py` script trains and saves the model.
  * **`notify/notify.py`** – Sends SMS notifications via Twilio or Amazon SNS when a transaction is flagged as potentially fraudulent.  It is used both to alert customers and to resend SMS if the initial notification failed.
  * **`admin/`** – Contains the Lambda functions exposed under `/admin` for the dashboard:
    * `list_transactions.py` – Lists all transactions for the table (supports pagination).
    * `get_transaction.py` – Fetches a transaction by ID.
    * `list_accounts.py` – Lists all accounts.
    * `get_account.py` and `update_account.py` – Retrieve or update account records (name, phone number, etc.).
  * **`user_response/`** – Processes a user’s SMS response to confirm whether a transaction is fraudulent or legitimate.  This updates the transaction record accordingly.
* **`lambdas/admin/fraud-admin-dashboard/`** – Front‑end React app for internal administrators.  It shows a list of transactions, allows resending SMS messages, displays account information, and includes basic metrics. Amplify and AWS API Gateway can be used to make authenticated requests.  Tailwind CSS provides styling.
* **Testing and deployment scripts** – `infrastructure/test_end_to_end_pipeline.py` runs an end‑to‑end test by seeding a transaction, invoking the scoring and notification functions, and verifying updates in DynamoDB.  Bash scripts like `deploy.sh`, `cleanup.sh` and `seed_db.py` facilitate deploying resources and seeding the database with sample data.

## What Works

* **Serverless backend** : The SAM template successfully deploys DynamoDB tables, Lambda functions and API Gateway resources.  The transaction pipeline triggers ML scoring, writes results back to DynamoDB, and sends SMS notifications through the `notify` Lambda.
* **Machine‑learning model** : A LightGBM model is trained using credit‑card fraud data.  The `score` Lambda loads the model and produces fraud probability scores.  Transactions with high risk are stored with `decision=alert` to drive notifications.
* **Admin dashboard** : The React app displays transactions and accounts with pagination and sorting.  Admins can resend SMS messages and edit account contact information.  The UI uses the same dark‑themed aesthetic as the project landing page.
* **Account management** : The `/admin/accounts` API and corresponding Lambdas support listing and updating accounts.  Editing an account updates the DynamoDB record and the dashboard reflects the change.

## What Doesn’t Work / Known Limitations

* **Large dataset not tracked in Git** : The training CSV (`fraudTest.csv`) is over GitHub’s 100 MB limit.  It must be downloaded [separately](https://www.kaggle.com/datasets/kartik2112/fraud-detection?resource=download) before training the model locally.
* **Auth** : The admin dashboard uses a hard‑coded password instead of Cognito or OAuth.
* **CORS and environment variables** : When running locally, API Gateway may reject requests due to missing CORS headers.  Ensure that `Access-Control-Allow-Origin` is set to `*` or your domain.  Environment variables such as table names and API endpoints must be set correctly in `.env` files.

## What’s Next / Future Work

* **Improve authentication** : Replace the hard‑coded admin password with proper authentication (e.g., AWS Cognito or SSO) and add authorization checks to the backend.
* **Enhance dashboard features** : Implement sorting, filtering and search for transactions.  Build a metrics dashboard that **visualizes** model performance, counts of alerts vs. legitimate transactions, and system latency.
* **Refine ML model** : Experiment with additional features (e.g., transaction metadata), feature engineering and hyperparameter tuning.  Evaluate performance on recent datasets and implement periodic retraining.
* **Error handling and validation** : Add robust input validation to the Lambdas, graceful error responses and logging.  Implement retries and dead‑letter queues for notification failures.
* **Support multi‑region deployment** : Parameterize the SAM template to deploy to multiple AWS regions and enable cross‑region failover.

This README provides an overview of the project’s structure, setup steps, working features and current limitations.  Feel free to contribute improvements via pull requests!
