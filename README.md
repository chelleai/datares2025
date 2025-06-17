# Chelle-AI

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

## Developing

### Collaboration

To make sure that each of you is able to work independently and without causing code conflicts amongst each other, it is best to develop on **branches**. A GitHub branch is a copy of the codebase to which you can push your own individual changes without impacting anyone else's work.

You should set up a new branch each time you want to try a new idea. Each branch should be connected to an issue, where you describe what you are going to do on the branch. For example, you might start by writing an issue describing a prompting strategy you want to try. Then you would check out a corresponding branch and implement that strategy. To set up a new branch:

1. Create an issue using the Issues board in the repository. Take note of the issue number after creating it.
2. Checkout a new branch named after the issue: `git checkout -b issue-[##]`, e.g. `git checkout -b issue-01`.
3. Develop and push your changes to your branch, using `git add`, `git commit`, and `git push`.

Typically in software projects, branches are creating with the intention of being merged back into main. Because the goal of these branches is to track experiments, we may instead choose to keep the branches and issues open for the lifetime of the project, as a record of experiments and to be able to easily flip between strategies by switching branches (via `git checkout [branch-name]`).

### Asset Processing Project

To make changes to the asset processing AI, you'll want to edit this file:

- `api/api/infrastructure/services/asset_processor.py`

There are two main methods in there to concern yourself with: `identify_terms`, and `synthesize_definition`. The docstrings explain more about what they do and how you should go about writing them.

### Guide Experience Project

To make changes to the guide AI, you'll want to edit this file:

` api/api/infrastructure/services/guide_agent.py`

There is one method in this file, `respond`, to focus on. The docstring explains more about what it does and how you should go about writing it.

## API Documentation

You can view the API docs at `http://localhost:8000/docs`. Here you can also interact with the API and invoke endpoints directly, if you want.

## Database

The application runs entirely on a local database, which will be stored at `db.json`. As the database is local, everyone has their own, so there is no need to worry about sharing data or stepping on each other's toes.

You can reset the database at any time simply by removing `db.json`. After you do, restart the API by killing the process and re-running `scripts/run-api`.
