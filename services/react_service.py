import logging

import pandas as pd
from llama_index.llms.openai import OpenAI
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.core.workflow import Context
from services.knowledge_service import KnowledgeService

knowledge_service = KnowledgeService(None)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def ask_knowledge_base(query: str) -> str:
    """
    This function queries the knowledge base with the given query and returns a synthesized response. If employee asks anything related to workplace policies, procedures, benefits, payroll or organizational regulations, this function must be invoked to get the accurate information from the knowledge base.      
    
    Args:
        query (str): The query must be interpreted to get relevant information from the knowledge base, rather than simply copy-pasting the employee query.

    Returns:
        str: Chunks retrieved from the knowledge base as a result of the employee query.
    """
    res = await knowledge_service.query(query, top_k=10)
    content = [chunk.content for chunk in res.chunks]

    return "\n\n".join(content)

def calculate_employee_working_hours(employee_email: str) -> str:
    """
    This function calculates the total working hours of an employee in the current time window of 14 days based on their ID. If the employee requests to know their current working hours or implies it in their query, the function must be invoked and calculate the hours accordingly.
    
    Args:
        employee_email (str): The email of the employee.

    Returns:
        str: A message indicating the total working hours of the employee so far against the expected hours, the remaining hours as well as the calculation strategy (normal 40 hrs/week or roster) in the current time window.
    """

    # Read docs/employee_timesheet.csv to get working hours data
    df = pd.read_csv("docs/employee_timesheet.csv")
    print(f"Calculating working hours for {employee_email}...")
    # Get total_working_hours and total_salary_hours columns between the current day and January 1, 2026 for the given employee_email
    expected_hours_per_month = 160
    employee_data = df[(df['employee_email'] == employee_email) & 
                       (pd.to_datetime(df['clock_in']) >= pd.to_datetime("2026-01-01 00:00:00"))]
    total_salary_hours = employee_data['total_salary_hours'].sum()
    remaining_hours = expected_hours_per_month - total_salary_hours
    calculation_strategy = "normal 40 hrs/week"

    response = (f"""Ажилчны ID: {employee_email}\n"
            f"Нийт цалингийн цаг: {total_salary_hours} hrs\n"
            f"Нийт ажиллах ёстой цаг: {expected_hours_per_month} hrs\n"
            f"Үлдсэн цаг: {remaining_hours} hrs\n"
            f"Тооцооллын стратеги: {calculation_strategy}""")
    
    print(f"Calculated working hours for {employee_email}: {response}")
    return response


def submit_timesheet(employee_email: str, timestamp: str, hours: float, absence_type: str) -> str:
    """
    Invoke this function if the employee wants to submit a timesheet entry for their absence or working hours. If they request you to submit a timesheet entry, but didn't provide all necessary details, you must inquire the timesheet entry details this function requires such as timestamp, hours and the reason for absence/work.
    
    Args:
        employee_email (str): The email of the employee.
        timestamp (str): The starting timestamp of the timesheet entry.
        hours (float): The number of hours worked or absent.
        absence_type (str): categorize the type of absence into one of the following list: "өвчтэй", "амралтаа авсан", "тасалсан", "хоцорсон", "гэрээс ажилласан" and "гадуур ажилласан".

    Returns:
        str: A message indicating the successful submission of the timesheet entry.
    """
    # Here you would implement the logic to submit the timesheet entry to your database or system.
    # For demonstration purposes, we'll just return a success message.
    response = (f"Ажилтан {employee_email}-ний хүсэлтийн дагуу ERP систем рүү дараах цаг бүртгэлийн хүсэлтийг илгээсэн:\n"
                f"Хэзээ: {timestamp}\n"
                f"Хэдэн цаг: {hours}\n"
                f"Ажил дээр байгаагүй шалтгаан: {absence_type}")
    
    print(f"Ажилтан {employee_email} дээр цаг бүртгэл илгээлээ: {response}")
    return response

workflow = FunctionAgent(
    tools=[calculate_employee_working_hours, ask_knowledge_base],
    llm=OpenAI(model="gpt-4o-mini", temperature=0.1),
    system_prompt="""
You are an assistant that helps employees with their queries regarding company policies, procedures, and personal work-related information. Employee queries are usually asked in Mongolian language.

First, you must extract the employee email and the intent of their query from the input. They may ask about various topics including leave policies, benefits, payroll, and working hours. 

Second, you must use the tools at your disposal and provide accurate and helpful information based on the employee's query and identified email.
     
For example, if an employee asks about their working hours or implies it in their query, you must call the 'calculate_employee_working_hours' function with their employee email to provide accurate information.

If the employee asks anything related to workplace policies, procedures, benefits, payroll or organizational regulations, you must call the 'ask_knowledge_base' function with their interpreted query to get the accurate information from the knowledge base. When querying the knowledge base, ensure that you interpret the employee's query to extract the intent rather than simply copy-pasting the employee query. If you respond according to the knowledge base citations, make sure to include their brief references in your final answer, such as according to [Citation 1], [Citation 2], etc.

If the employee requests to submit a timesheet entry for their absence or working hours, you must ask for the necessary details such as timestamp, hours, and absence type. Once you have all the required information, you should inform the employee that their timesheet entry has been submitted successfully. However, you do not need to implement the actual submission logic; just confirm the submission in your response.

Try to format your final response in a clear and structured manner, using bullet points or numbered lists where appropriate to enhance readability.
"""
)

ctx = Context(workflow)