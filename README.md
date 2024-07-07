# historypy
chm history in python

**Computer History Bot - Richard Sawey**

A chatbot for research on computing history, a RAG implementation on selections from the archives at the Computer History Museum, 
Mountain View, https://www.computerhistory.org

        Thank you for trying HistoryBot, a prototype chat bot designed to provide an interactive
        experience for those researching or interested in computer history. In History Bot's current
        experimental configuration the only source document used is a Computer History Museum docent's
        script. The author wrote this script as part of his responsibilities as a museum docent and the
        script includes details on a range of important computers from history such as ENIAC and the 
        IBM 360.

        With funding I'd look to expand the History Bot to include the many personal histories collected
        by the museum and currently archived in either PDF or video format. 

        This script needs three input arguments, first the LLM to use (right now
        this is ignored, I always use gpt-4-turbo-preview). Next a 0 or 1 is required
        to indicate if we should use the existing vector stores. A '1' says use the persisted vector store 
        and '0' tells History Bot to rebuild the vector indexes. 
        
        This script also assumes the existence of an environment variable OPENAI_API_KEY
        that obviously contains your OpenAI key for the LLM.
        
        In command line script mode just enter:
        python3 chmbotv7.py chatgpt 1

        Richard Sawey TECH16 March 2024

