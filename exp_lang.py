from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import ChatPromptTemplate
from langchain.agents import tool, OpenAIFunctionsAgent, AgentExecutor
from langchain.prompts.chat import SystemMessagePromptTemplate
from langchain import PromptTemplate

#Figure and Question
figure = "Charles Darwin"
question = "Why did you go to the Galapagos Islands?"

llm = ChatOpenAI(model_name="gpt-4", temperature=1)


################################## INDIVIDUAL CHAINS ##################################

#Judge Chain, judges whether a question is appropriate for a school setting or not
judge_template = PromptTemplate(
    input_variables=["question"],
    template="Judge if the following question is appropriate for a school setting or not. Appropriate questions are anything that may be related to a school course or a general wonderings. Innapropriate questions have absolutely nothing to do with education, or may include swear words. If it is appropriate, write 'appropriate', if not, write 'innapropriate'.\n\n Question: {question}"
)
judge_prompt = ChatPromptTemplate.from_messages([SystemMessagePromptTemplate
                                                 (prompt=judge_template)])                                                  
judge_chain = LLMChain(llm=llm, prompt=judge_template)

#Response Chain, responds to a question in a historically accurate, but also light and humorous way
response_template = PromptTemplate(
    input_variables=["figure", "question"],
    template="You are {figure}. Answer the following question in a educationally accurate, but also light and humorous way. \n\nQuestion: {question}\n\nAnswer:"
)
response_prompt = ChatPromptTemplate.from_messages([SystemMessagePromptTemplate(prompt=response_template)])
response_chain = LLMChain(llm=llm, prompt=response_template)

#Excuse Chain, makes up an excuse to not answer a question
excuse_template = PromptTemplate(
    input_variables=["figure"],
    template="You are {figure}. Make up a convenient one-sentence excuse to not answer a student's question in a historically accurate and lighthearted way. \n\nExcuse:"
)
excuse_prompt = ChatPromptTemplate.from_messages([SystemMessagePromptTemplate(prompt=excuse_template)])
excuse_chain = LLMChain(llm=llm, prompt=excuse_template)


################################## AGENT ##################################

#Tools. Functions used by the agent which decides when to run each chain (run each prompt)
@tool
def run_judge_chain(question: str) -> str:
    """Called to determine if an llm determined the Question as appropriate or not"""
    return judge_chain.run(question=question)

@tool
def run_response_chain(figure: str, question: str) -> str:
    """Called to get an llm response to a question"""
    return response_chain.run(figure=figure, question=question)

@tool
def run_excuse_chain(figure: str) -> str:
    """Called to get an llm excuse to not answer a question"""
    return excuse_chain.run(figure=figure)

tools = [run_judge_chain, run_response_chain, run_excuse_chain]

#System message which is also the prompt for the agent
system_message = PromptTemplate(
    input_variables=["figure", "question"],
    template="You are given a Figure and a Question below. You must decide if the if the question is appropriate for a school setting or not by running the judge chain. If appropriate then call the response chain. Call the excuse chain regardless. Format your response as a json object with two fields: \"excuse\" and \"response\". Leave the response field empty if you did not call it. \n\nFigure: {figure}\n\nQuestion: {question}")
system_message = SystemMessagePromptTemplate(prompt=system_message)
prompt = OpenAIFunctionsAgent.create_prompt(system_message=system_message.format(figure=figure, question=question))

agent = OpenAIFunctionsAgent(llm=llm, prompt=prompt, tools=tools)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

print(agent_executor.run("run")) #Have to provide an argument to run(), just a workaround. Probably makes more sense to paste the system message in here