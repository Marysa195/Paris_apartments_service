## Project description

![](https://github.com/Marysa195/Paris_apartments_service/blob/main/demo/%D0%9F%D1%80%D0%B5%D0%B7%D0%B5%D0%BD%D1%82%D0%B0%D1%86%D0%B8%D1%8F.gif)

The goal of this pet-project is to create a web-service which will use
ML-model that can predict the cost of an apartment in Paris, based on the
information provided. 
Therefore, in order to save time, I will allow myself to simplify the
preprocessing steps and EDA: to use coarse filling in the gaps and removing
uninformative and poorly filled features.

More detailed analyzes of dataset features and ML-model building were
carried out by me in the first pet project, which can be found at the attached
link.

The data for this project was obtained by parsing the site www.bienici.com - a
French site for finding real estate for buying / renting apartments in France.
My request was only for apartments to buy in Paris. The data was obtained by 
requesting the site's hidden API, so it contains a large number of features 
that are not informative for my project (about the parameters for displaying
an ad on the site for example) that require filtering.

## Main services used in pet-project
- Jupyter Notebook - for data exploration and initial hypothesis testing;
- PyCharm - for module development;
- FastApi - for adding to code backend part;
- Streamlit - for adding to code frontend part;
- Docker compose - for project composing.

## Folders
- `/backend` - Folder with FastAPI project;
- `/frontend` - Folder with Streamlit project;
- `/config` - Folder with the configuration file;
- `/data` - Folder containing raw data, processed data, unique values in JSON
format, as well as an unlabeled file for submission to the model input;
- `/demo` - Folder containing a demo of the service in Streamlit UI in gif
format;
- `/models` - Folder containing the saved model after training, as well as the
study object (Optuna);
- `/notebooks` - Folder containing jupyter notebooks with preliminary data
analysis;
- `/report` - Folder containing information about the best parameters and
metrics after training.

## Project launch

- Building services from images inside backend/frontend and running containers
offline

`docker compose up -d`

- Service stop 

`docker compose stop`

- Delete **stopped** containers

`docker compose rm`
