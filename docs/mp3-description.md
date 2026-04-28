## Mini-Project 3: Predictive Modeling and Optimization for Real Estate Investment

You are hired as a **Machine Learning and Optimization Consulting Team** for a real estate investment company that wants to make better property investment decisions using data.

The company is interested in:

- Predicting the market value of houses
- Identifying which properties are the most promising for investment
- Allocating a limited renovation budget effectively

However, leadership does not want only a prediction model. They want a system that combines **prediction and decision-making**. Your team’s job is to build a workflow that uses machine learning to estimate housing prices, then uses optimization techniques to decide which houses should be selected under budget constraints

---

## Dataset

This project will use the **Ames Housing Dataset**.

Dataset link: [https://www.kaggle.com/datasets/prevek18/ames-housing-dataset Links to an external site.](https://www.kaggle.com/datasets/prevek18/ames-housing-dataset)

You may use a cleaned version of the dataset if needed, but you must clearly state any preprocessing steps your team performed.

Your team will address the following project question:

**How can a real estate company use house features to predict property value and make better investment decisions under a limited renovation budget?**

Your project must include three connected components:

1. **Neural Network** for predicting house prices
2. **Linear Programming** for selecting houses under a fixed budget

---

## Part 1: Feature Selection and Problem Framing

At the start of your analysis, your team must select **six to eight input variables** from the dataset that you believe are most relevant for predicting house price. Your selection must be intentional and justified. Do not include every variable without explanation.

For each selected variable, briefly explain:

- What the variable represents
- Why you believe it may influence house prices

You must also:

- Identify the **target variable**
- explain why this is a **regression problem** rather than a classification problem

---

## Part 2: Data Preparation and Baseline

Before building your neural network, your team must prepare the data appropriately.

Your preprocessing should include, where needed:

- handling missing values
- encoding categorical variables
- scaling or normalizing numerical variables
- splitting the data into training and testing sets

You must also define:

- an appropriate evaluation metric for the prediction task
- a simple baseline for comparison

Examples of suitable regression metrics include:

- RMSE
- MAE
- R²

Briefly justify why your selected metric is appropriate.

---

## Part 3: Neural Network Modeling

Build a **neural network** to predict house prices.

Your implementation must include:

- at least **two different neural network architectures**
- variation in the number of hidden layers or neurons

Your team must report:

- performance on the test set
- comparison between models

You should briefly discuss:

- Which architecture performed better
- whether increasing complexity helped
- any signs of overfitting or underfitting

---

## Part 4: Optimization

Now assume your company has a **limited renovation budget** and cannot invest in every house. ASs a team, you need to formulate and solve an **optimization problem** to select a subset of houses that maximizes total profit under constraints.

Your team must:

- Define **decision variables** (e.g., whether to select a house)
- Define an **objective function** (maximize total profit)
- Include at least one **constraint** (e.g., budget limit)

Example objective: $Maximize \; \sum (profit_i \cdot x_i)$

**Briefly explain:**

- Why your solution makes sense
- How constraints affected your decisions

---

## Instructions:

#### Team Requirements

- Team size: You will work in teams of 3.

#### Team Contract (Required)

Each team must submit a **2-page contract** including:
- **Team members and roles**
- **Component ownership** (each member must lead one):
	- Neural Network (prediction)
		- Optimization
		- Data preprocessing/report (if needed)
		- Presentation
		- For each member, clearly state:
		- which component they are leading
				- what their responsibilities are
- **Communication plan**
- **Meeting schedule**
- **Conflict resolution strategy**

---

## Submission Requirements

Your team will submit:

***1) Jupyter Notebook***

- all code (preprocessing, modeling, optimization)
- visualizations
- clear comments explaining decisions

***2) Short Report (2–3 pages)***

Include:

- problem framing
- feature selection justification
- model comparison
- optimization setup
- key insights and limitations

***3) Team Contract***

***4) Presentation: Your team must deliver a final presentation in the last week of class summarizing their approach, results, and key insights.***

***5) Peer Evaluation (Individual Submission)***

Each student must submit:

- Contribution (%) of each team member
- Short reflection (2–3 sentences), including:
	- Did each member effectively lead their assigned component?
		- Any issues in collaboration?

---

## Submission Summary

- **One team submission:**
	- Jupyter Notebook
		- Short report
		- Team contract
- **Individual submission (per student):**
	- Peer evaluation

**Grading Rubric (Total: 100 Points)**

| Component | Criteria | Points |
| --- | --- | --- |
| **Neural Network (Prediction)** | Correct implementation | 10 |
|  | Use of at least 2 architectures & comparison | 8 |
|  | Appropriate evaluation metrics | 6 |
|  | Interpretation (overfitting, performance) | 6 |
| **Subtotal** |  | **30** |
| **Data Preparation & Feature Selection** | Thoughtful feature selection (4–6 features) | 5 |
|  | Justification of selected features | 5 |
|  | Proper preprocessing (encoding, scaling, missing values) | 5 |
| **Subtotal** |  | **15** |
| **Optimization** | Clear formulation (variables, objective, constraints) | 10 |
|  | Correct implementation | 8 |
|  | Logical results (selection, cost, profit) | 4 |
|  | Interpretation of decisions | 3 |
| **Subtotal** |  | **25** |
| **Report (2–3 pages)** | Problem framing & clarity | 5 |
|  | Explanation of modeling & optimization | 5 |
|  | Insights & limitations | 5 |
| **Subtotal** |  | **15** |
| **Presentation** | Clear explanation of workflow | 4 |
|  | Understanding of methods | 3 |
|  | Q&A / clarity | 3 |
| **Subtotal** |  | **10** |
| **Team Contract** | Roles & component ownership | 2 |
|  | Defined responsibilities | 2 |
|  | Communication & conflict plan | 1 |
| **Subtotal** |  | **5** |
| **Peer Evaluation (Individual)** | Submitted and complete (all required components addressed) | **Pass/Fail (Required)** |
| **Total** |  | **100** |