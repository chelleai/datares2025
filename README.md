# Chelle-DataRes 2025

Welcome, DataRes students!

The goals of this project are two-fold. First, we want to optimize the process of extracting key educational terminology and citations from source materials, such as papers, blogs, and textbook chapters. Second, we want to improve the learning experience of communicating with an intelligent AI tutor.

To facilitate working on both of these projects, this repository contains an API and small web app for doing just these things. In this app you can:

1. Upload and process Markdown documents to produce collections of "Concepts", where a Concept is a term with a set of citations and a synthesized definition.
2. Chat with a "Guide" that has awareness of a set of Concepts and a student's learning style.

## Getting Started

After cloning this repository, you'll need to:

1. Install [uv](https://docs.astral.sh/uv/getting-started/installation/).
2. Run the initialization script by running: `scripts/init`
3. Start the API in one process by running: `scripts/run-api`
4. Start the app in another process by running: `scripts/run-app`

At this point, you should have a running application at `http://localhost:3000` that can interact with the backend, which is running locally at `http://localhost:8000`.

## API Documentation

You can view the API docs at `http://localhost:8000/docs`. Here you can also interact with the API and invoke endpoints directly, if you want.

## Database

The application runs entirely on a local database, which will be stored at `db.json`. As the database is local, everyone has their own, so there is no need to worry about sharing data or stepping on each other's toes.

You can reset the database at any time simply by removing `db.json`. After you do, restart the API by killing the process and re-running `scripts/run-api`.