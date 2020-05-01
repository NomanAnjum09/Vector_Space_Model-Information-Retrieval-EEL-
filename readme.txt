              With The Name Of Allah Most Beneficial And Most Mercuiful
              
                This is Vector Space Model for Information Retrieval


How To Run:

1) Install python3
2) Install pipenv
3) Open Assignment Folder
4) Type 'pipenv install'
5) Type 'pipenv shell'

To run simple tf*idf model:
    Type 'python main.py'

To run tf*idf model with unit vector optimization":
    Type 'python unit.py'

To verify unit vector optimization:
    Type 'practice.py'


Note:
    While Switching From One Model to Other Run From Scratch To Rebuild Indexes
    
Future Work:
    Generation of indexes(ie:documentFrequency,Term Frequency and Vectors) are written on different file for each document.
    Using Json object for content of each file and saving all objects on a single file will be more efficient in term of performance as only 1 IO will be required.
