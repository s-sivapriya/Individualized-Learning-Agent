Building Individualized Learning Agent with Autogen
Installation
To run this agent, you will need to install the following packages:
pip install “autogen-ext[openai,azure]”
pip install azure-ai-generative
pip install PyPDF2

To use other model providers, you will need to install a different extra for the autogen-ext package. See the Models documentation for more information.
Running the Agent Team with UserProxyAgent
Download the Sample Files Begin by downloading the "Sample IEPs" and "Textbook Chapters" folders. These contain example documents sourced from publicly available educational resources.
Launch the Agent Run the application in your terminal (Individualized Learning Agent.py). Upon execution, the system will prompt you to enter the file path to a student’s Individualized Education Plan (IEP).
Provide a Sample IEP Enter the path to a sample document from the "Sample IEPs" folder. The agent will analyze the IEP to determine the student’s reading level.
Summarize Educational Content Once a valid reading level is extracted, the second agent is activated. You will then be prompted to provide the path to a PDF document (e.g., a textbook chapter) to be summarized according to the student’s identified reading level.
Access the Output The summary is generated and saved as a text file in the same directory as the original PDF document, making it easy to locate and use alongside the source material.
Note: Please ensure that the agent has write permissions to the folder containing the lessons
