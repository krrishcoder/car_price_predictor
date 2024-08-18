

# this sets up the container with python 3.9 installed
FROM python:3.9

# this copies everything in your current dire. to the /app dir in the container
COPY . /app

# this sets the /app direct. as the working dire. for
# any RUN , CMD , ENTRYPOINT , or COPY instructions that follow.

WORKDIR /app

# this runs app install for all the packages listed in your requirement.txt file
RUN pip install -r requirement.txt


# this tells Docker to listen on port at runtime . PORT 80 is the standard port for HTTP
EXPOSE 80

# This command creates a .streamlit directory in the home direct. of the container.

RUN mkdir ~/.streamlit

# this copies your streamlit config. file into the .streamlit direct. you just created
RUN cp config.toml ~/.streamlit/config.toml

RUN cp credentials.toml ~/.streamlit/credentials.toml


# this sets the default command for the container to run the app with streamlit

ENTRYPOINT ["streamlit", "run"]

CMD ["app.py"]


