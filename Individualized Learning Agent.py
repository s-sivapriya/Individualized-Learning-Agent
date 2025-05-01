
from typing import List, Sequence
from autogen_agentchat.agents import AssistantAgent, UserProxyAgent, BaseChatAgent
from autogen_agentchat.conditions import MaxMessageTermination, TextMentionTermination
from autogen_agentchat.messages import BaseAgentEvent, BaseChatMessage, TextMessage
from autogen_agentchat.teams import SelectorGroupChat, RoundRobinGroupChat
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import AzureOpenAIChatCompletionClient
from autogen_agentchat.base import Response
from autogen_core import CancellationToken
from autogen_core.models import UserMessage
from PyPDF2 import PdfReader

import fitz  # PyMuPDF for reading PDF
import asyncio
import os


iep_reading_summary = None #Stores the summary generated


#model client for Assistant Agents
model_client = AzureOpenAIChatCompletionClient(
        azure_deployment="gpt-4o",
        model="gpt-4o",
        api_version="2024-12-01-preview",
        azure_endpoint= "https://<your-endpoint>.openai.azure.com/",
        api_key="<your-key>", 
    )

#Function to read a pdf
async def read_pdf_text(file_path) ->str:
    # read a pdf file and if there is no text in the file, return None
    if not os.path.exists(file_path):
        print("File does not exist.")
        return None
    try:
        with open(file_path, 'rb') as f:
            header = f.read(5)
            if header != b'%PDF-':
                print("Not a valid PDF.")
                return None
            
        reader = PdfReader(file_path)
        _ = reader.pages[0]  # Try accessing a page
        
        text = ""
        
        for page in reader.pages:
            text += page.extract_text() or ''
        
        return text
    except Exception as e:
        print(f"PDF validation failed: {e}")
        return None
    

    

# Function to summarize using the IEP document to get the reading level of the student
async def summarize_reading_level_with_llm(text:str)->str:
    global iep_reading_summary
    print("Gathering the student information ...")

    prompt = f"""This is the information about a student provided by their special education teacher. 
    Gather the student's reading and understanding level, grade level from the following text:\n\n{text}
    If the text does not contain any information about the reading level, grade level,
    or scores, please respond with "No information found"."""

    response = await model_client.create([UserMessage(content=(prompt), source="user")])
    
    #If the given pdf file does not have any reading level information, return None
    if response.content == "No information found.": 
        iep_reading_summary = None
        return None
    else:
        iep_reading_summary = response.content
    
    
    return iep_reading_summary


# Function to summarize using Azure OpenAI
async def summarize_text_with_llm(text:str)->str:
    #Summarize the text in a given document to make understandable based on the reading level of the student
    print("Summarizing the document ...")
    prompt = F"""Redact the following text:\n\n{text}, retain as much information as possible.
    but make it understandable based on the stdents grade and reading level from: \n\n{iep_reading_summary}"""

    response = await model_client.create([UserMessage(content=prompt, source="user")])
    return response.content


#Define a function to save summary
async def save_summary_to_file(summary, filename="summary.txt"):
    print("Saving the summary ...")
    try: 
        with open(filename, "w") as f:
            f.write(summary)
            return f"Summary saved to {filename}"
    except Exception as e:
        print(f"Error saving summary: {e}")
        return None



# Create the agent to read IEP documents
class IepReadingAgent(BaseChatAgent):
    def __init__(self, name: str, description: str) -> None:
        
        super().__init__(name, description=description)
        self._message_history: List[BaseChatMessage] = []

    @property
    def produced_message_types(self) -> Sequence[type[BaseChatMessage]]:
        return (TextMessage,)

    async def on_messages(self, messages: Sequence[BaseChatMessage], cancellation_token: CancellationToken) -> Response:
        global iep_reading_summary
        # Update the message history.
        # NOTE: it is possible the messages is an empty list, which means the agent was selected previously.
        self._message_history.extend(messages)
        
        # Parse the last message.
        assert isinstance(self._message_history[-1], TextMessage)
        #summarize the file
        text_doc = await read_pdf_text(self._message_history[-1].content)
        
        if text_doc is None:
            response_message = TextMessage(content=str("Enter a valid IEP file"), source=self.name)
        else:
            
            iep_summary = await summarize_reading_level_with_llm(text_doc)
            
            if iep_summary is None:
                response_message = TextMessage(content=str("Unable to summarize the file"), source=self.name)
            else:
                response_message = TextMessage(content=str("TERMINATE"), source=self.name)

        
        self._message_history.append(response_message)
        # Return the response.
        
        return Response(chat_message=response_message)

    async def on_reset(self, cancellation_token: CancellationToken) -> None:
        pass


# Create the agent to summarize the given document
class TextSummarizingAgent(BaseChatAgent):
    def __init__(self, name: str, description: str) -> None:
       
        super().__init__(name, description=description)
        self._message_history: List[BaseChatMessage] = []

    @property
    def produced_message_types(self) -> Sequence[type[BaseChatMessage]]:
        return (TextMessage,)

    async def on_messages(self, messages: Sequence[BaseChatMessage], cancellation_token: CancellationToken) -> Response:
        # Update the message history.
        # NOTE: it is possible the messages is an empty list, which means the agent was selected previously.
        self._message_history.extend(messages)
        
        # Parse the last message.
        assert isinstance(self._message_history[-1], TextMessage)
        #read the file
        file_name = self._message_history[-1].content
        text_doc = await read_pdf_text(file_name)
        
        if text_doc is None:
            response_message = TextMessage(content=str("Enter a valid document to summarize"), source=self.name)
        else:
            # Process the Document, to get the summary
            document_summary = await summarize_text_with_llm(text_doc)
            
            #Save the summary
            file_name = file_name.split(".")[-2] + "_summary.txt"
            await save_summary_to_file(document_summary, file_name)
            response_message = TextMessage(content=str(f"Summary saved to \n\n{file_name}"), source=self.name)
            
           
        self._message_history.append(response_message)
        # Return the response.
        
        return Response(chat_message=response_message)

    async def on_reset(self, cancellation_token: CancellationToken) -> None:
        pass


async def main():
    iep_agent = IepReadingAgent("iep_reading_agent", "Reads the IEP documents provided by user.")
    summarizer_agent = TextSummarizingAgent("text_summarizing_agent", "Summarizes a given pdf link documents provided by user.")
    user_proxy = UserProxyAgent(
        name="user_proxy",
        input_func = input,
    )
    
    global iep_reading_summary
    
    #Create a Round Robin team of User_proxy and IEP agent to get IEP information, 
    # this will terminate when (1) user gives a valid document (2) user enters "TERMINATE" or (3) the number of messages exceeds 8
    
    print("Please enter the Individualized Education Plan (IEP) document link or enter TERMINATE to stop the process:")
    text_mention_termination = TextMentionTermination("TERMINATE")
    max_messages_termination = MaxMessageTermination(max_messages=8)
    termination = text_mention_termination | max_messages_termination

    iep_team = RoundRobinGroupChat([user_proxy,iep_agent], termination_condition=termination)
    stream = iep_team.run_stream(task="Enter the IEP document link")
    await Console(stream)

    if iep_reading_summary is None:
        print("No reading level information found in the IEP document.")
        return
    
    #Create a Round Robin team of User_proxy and IEP agent to get IEP information,
    #Create a Round Robin team of User_proxy and summarizer agent to get the summary of the documents entered by user and save them
    #This will terminate when (1) User enters TERMINATE (2) the number of messages exceeds 10
    #NOTE: The summary will be saved in the same directory as the script with the name <document_name>_summary.txt
    text_mention_termination = TextMentionTermination("TERMINATE")
    max_messages_termination = MaxMessageTermination(max_messages=10)
    termination = text_mention_termination | max_messages_termination
    
    print("Please enter the pdf document link you would like to read or enter TERMINATE to stop the process:")
    summerizer_team = RoundRobinGroupChat([user_proxy,summarizer_agent], termination_condition=termination)
    stream = summerizer_team.run_stream(task="Enter the document link to summarize")
    await Console(stream)



# Step 6: Start the flow
if __name__ == "__main__":
   asyncio.run(main())