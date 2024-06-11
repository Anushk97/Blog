---
layout: post
title:  "Get your facts in order 🐥: An application to record decisions made during meetings"
date:   2024-05-09 12:54:15 +0800
categories: jekyll update
---

This app was part of a job interview assignment I worked on. The idea seem pretty cool so I kept working on it a little longer, upgraded the frontend - learnt some javascript, html and flask in the process. 

The project was a good learning about fundamental data structures, APIs, and cloud deployment. The app is deployed [here](https://llm-bot-pficsoahpa-uc.a.run.app/).

## The Ask 💭 
I was asked to create an application which can use LLM to summarize call logs in a nutshell. The app should be able to do 2 things:

1. Submit question and get summary for call logs
- The call logs where in form of a url which contained the date it was recorded on and generally had a product design decision made during the call. 
- In case there is a contradiction between decisions made on 2 different days, the application should be able to recognize that and ask the user to keep the previous fact or delete it. 

2. Show the facts on a timeline 
- The application should show each one fact/decision from the call log for that date in a timeline manner.
- The user should also be able to delete previous facts made in the past.

**call log example**
```
1
00:01:11,430 --> 00:01:40,520
John: Hello, everybody. Let's start with the product design discussion. I think we should go with a modular design for our product. It will allow us to easily add or remove features as needed.

2
00:01:41,450 --> 00:01:49,190
Sara: I agree with John. A modular design will provide us with the flexibility we need. Also, I suggest we use a responsive design to ensure our product works well on all devices. Finally, I think we should use websockets to improve latency and provide real-time updates.

3
00:01:49,340 --> 00:01:50,040
Mike: Sounds good to me. I also propose we use a dark theme for the user interface. It's trendy and reduces eye strain for users. Let's hold off on the websockets for now since it's a little bit too much work.
```

## Backend and tech stack ⛺
So thinking about these features I need to have 2 pages - first to submit question and document URL and second to view facts. 

### Submit question and document logic
To start with, I need to create an POST and GET API to submit URL, read and process it and store the summary in the backend. For this I choose Flask.

In the flask application, it is crucial to store the data in a dictionary. This will help retrieve and store it effectively.

This dictionary will store the question, document url, extracted facts by day and the status of the request.

```
questions_and_facts = {
    "question": [],
    "documents": [],
    "factsByDay": {},
    "status": "processing" 
}
```
Now let's create the first method of submitting the url. 
- In it we will first extract the text from the document URLs and the date mentioned in the URL. 
- Then we will use OpenAI 3.5 turbo to extract key summary points from that document. 
- Get the current timestamp of the document and format it
- Append the document URL to our questions_and_facts dictionary
- Keep a list of existing facts in the dictionary which we can look up for contradictions.
- Append the summary, timestamp, question, document and contradiction in the factsByDay dictionary inside the questions_and_facts dictionary.
- Keep a list of suggestions which we can show on the front end everytime the submits a document. 
- Append the question and status to the questions_and_facts dictionary 
- return suggestions 

```
@app.route('/submit_question_and_documents', methods = ['POST'])
def submit_question_and_documents():
    
    data = request.get_json()
    question = data.get('question', '')
    document_urls = data.get('documents', [])
    
    document_texts = fetch_and_extract_text(document_urls)
    all_suggestions = {}
    
    for url, document_text in document_texts.items():
        date_match = re.search(r'\d{4}\d{2}\d{2}', url)
        
        if date_match:
            document_date = date_match.group(0)
        else:
            document_date = "2024-01-02"  # Default date
            
        document_date_obj = datetime.strptime(document_date, "%Y%m%d")
        
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo-0125",
            messages=[
            {"role": "system", "content": """
            You are a smart assistant tasked with summarizing key decisions and important facts from a series of call logs.\n
            Summarize each new point or insight or decision made in the call log.\n
            Start each point by the team has.\n 
            Generate at least three points.\n
            """
            },
                {"role": "user", "content": f"{document_text}"},
                {"role": "system", "content": f"Based on the above, {question}"}
    ],
            n=1,
            stop=None,
            temperature=0.4
        )
        
        extracted_facts = response.choices[0].message.content.strip().split('\n')
        
        current_timestamp = datetime.now().isoformat()
        formatted_date = document_date_obj.strftime("%Y-%m-%d")
        questions_and_facts['documents'].append(url)
        if formatted_date in all_suggestions:
            pass
        else:
            all_suggestions[formatted_date] = []

        existing_facts = [fact['text'] for facts_list in questions_and_facts["factsByDay"].values() for fact in facts_list]

        for fact_text in extracted_facts:
            if fact_text.strip():
                contradictions = find_contradictions(fact_text, existing_facts)
                fact_detail = {"text": fact_text, "timestamp": current_timestamp, "question": question, "documents":[url], "contradictions":contradictions}
                
                questions_and_facts["factsByDay"].setdefault(formatted_date, []).append(fact_detail)            
                
                all_suggestions[formatted_date].append(fact_detail)
    
    questions_and_facts['question'].append(question)
    questions_and_facts["status"] = "done"
    
    return jsonify({'suggestions': all_suggestions})
```

#### Helper functions
I used two helper functions in the code above, that is to extract text from the documents and find contradictions.
- The first one uses the requests library to access the URL and extract text from it. 
- The second function initiates a contradiction dictionary 
    - takes a new fact and list of existing facts 
    - Iterates over existing facts and uses OpenAI API to compare it with the new fact. 
    - If it is the same, then it appends the fact in the contradictions dictionary.

```
def fetch_and_extract_text(urls):
    extracted_texts = {}
    for url in urls:
        if not url or not (url.startswith('http://') or url.startswith('https://')):
            print(f"Skipping invalid URL: '{url}'")
        try:
            response = requests.get(url)
            response.raise_for_status()  # Raises an HTTPError for bad responses
            extracted_texts[url] = response.text
        except requests.exceptions.RequestException as e:
            print(f"Failed to fetch or extract text from {url}: {e}")
            extracted_texts[url] = "" 
    return extracted_texts

def find_contradictions(new_fact, existing_facts):
    contradictions = {'existing_facts': [],
                    'status': 'keep'}
    for existing_fact in existing_facts:
        prompt = f"Does the statement: '{new_fact}' contradict the existing fact: '{existing_fact}'?"

        response = openai.chat.completions.create(
            model="gpt-3.5-turbo-0125",
            messages=[
            {"role": "system", "content": prompt}],
        n=1,
        stop=None,
        temperature=0.4
        )
        
        if response.choices:
            content = response.choices[0].message.content
            if "yes" in content.lower():
                contradictions['existing_facts'].append(existing_fact)
    
    return contradictions
```



### View Facts logic
In the view facts logic, we need to do the following:
- Show the facts by date on a timeline. 
- For that we need to read from questions_and_facts dictionary, iterate over the facts, find minimum date from the timestamp to show the first fact
- Maintain a list of unique questions to be used to lookup facts later.
- Becuase we plan to filter by date we need to have a start date.
- Return the unique question, start date, questions_and_facts dictionary and current time for frontend.

```
@app.route('/view_facts', methods=['GET'])
def view_facts_page():
    
    filtered_questions_and_facts = copy.deepcopy(questions_and_facts)
    current_time = datetime.now()
    question = questions_and_facts.get("question")
    min_date = datetime.max
    unique_questions = list(set(questions_and_facts["question"]))
    
    for date, facts in filtered_questions_and_facts["factsByDay"].items():
        for fact in facts:
            if 'timestamp' in fact:
                fact_date = datetime.strptime(fact['timestamp'].split("T")[0], "%Y-%m-%d")
                if fact_date < min_date:
                    min_date = fact_date
                hours_diff = (current_time - fact_date).total_seconds() / 3600
                fact['hours_diff'] = hours_diff
    
    if filtered_questions_and_facts["factsByDay"]:
    
        all_dates = list(questions_and_facts["factsByDay"].keys())
        start_date = min(all_dates)  # Find the earliest date
    else:
        start_date = min_date.strftime('%Y-%m-%d')
    
    return render_template('view_facts.html',unique_questions=unique_questions, start_date=start_date, questions_and_facts=filtered_questions_and_facts, current_time=current_time)
```

## FrontEnd 🚦

### Submit question and documents page
Lets start with the submit question and documents page. 
First to set up the page architecture, we have to write in HTML and Javascript. First we create a form with a textbox which will take comma seperated URLs and a submit button.

```
<link rel="stylesheet" href="path/to/your/styles.css">
<html lang="en" dir="ltr">
<div class="container"></div>
<head>
    <title>Submit Question and Documents</title>
</head>
<body>
    <div class="container">
        <h1>Submit Your Question and Document URLs</h1>
        <form action="/submit_question_and_documents" method="post">
            <label for="question">Question:</label><br>
            <input type="text" id="question" name="question"><br>
            <label for="documents">Document URLs (comma-separated):</label><br>
            <textarea id="documents" name="documents" rows="5" cols="33"></textarea><br>
            <input type="button" value="submit" onclick="submitForm()" class="submit-button">
        </form>
    </div>

    <div id="suggestionsModal" style="display:none;">
        <div id="suggestionsContent">
        </div>
        
    </div>
    
    <div id="loadingIndicator" style="display:none;">
        <div class="loader"></div>
        <p>Please wait...</p>
    </div>
```

Next we define a javascript function trigger our backend logic when the submit form is clicked. In this we send the question and document URLs to the submit_question_and_documents function defined earlier.

```
 <script>
        var suggestions; 
        function submitForm() {
            var question = document.getElementById('question').value;
            var documentsInput = document.getElementById('documents').value;
            var documents = documentsInput.split(',').map(doc => doc.trim());
            
            const invalidUrls = documents.filter(url => !isValidUrl(url));

            if (invalidUrls.length > 0) {
                // Display error message for invalid URLs
                alert("The following URLs are invalid: " + invalidUrls.join(', '));
                return; // Stop the function execution here
            }

            document.getElementById('loadingIndicator').style.display = 'block';

            fetch('/submit_question_and_documents', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    question: question,
                    documents: documents
                })
            })
            .then(response => response.json())
            .then(data => {
                console.log(data);
                displaySuggestions(data.suggestions);
            });
        }
```
We also need to make sure that the URLs are valid. This function does a simple check of whether the URL has http or https protocol in it.
```
function isValidUrl(urlString) {
            let url;

            try {
                url = new URL(urlString);
            } catch (_) {
                return false;
            }

            return url.protocol === "http:" || url.protocol === "https:";
        }
```
Next we want to display the suggestions in a pop up box with accept or reject buttons. 
- This function reads the suggestions from the backend and display contradictions in red (if any) and provide an option to keep or remove them. 
- It also gives an option to accept or reject the suggestions. 
- If rejected, the suggestions will not be shown on the view facts page. 
- Finally, it also has accept all and reject all buttons which will do all operation in one click.

```
function displaySuggestions(suggestions) {
            document.getElementById('loadingIndicator').style.display = 'none';

            var modalContent = document.getElementById('suggestionsContent');
            modalContent.innerHTML = '';
            
            window.suggestions = suggestions; 

            Object.keys(suggestions).forEach(function(date, index) {
                
                var dateSection = document.createElement("div");
                var dateHeader = document.createElement("h4");
                dateHeader.textContent = "Date: " + date;
                dateSection.appendChild(dateHeader);
                
                suggestions[date].forEach(function(suggestion, index) {
                    
                    var suggestionDiv = document.createElement("div");
                    var textPara = document.createElement("p");
                    console.log("text", textPara);
                    textPara.textContent = suggestion.text;
                    suggestionDiv.appendChild(textPara);

                    if (suggestion.contradictions && suggestion.contradictions["existing_facts"].length > 0) {
                        var contradictionsPara = document.createElement("p");
                        console.log("contradiction", contradictionsPara);
                        contradictionsPara.textContent = "Contradictions:";
                        contradictionsPara.style.color = "red";
                        
                        suggestion.contradictions['existing_facts'].forEach(function(contradiction, contradictionIndex) {
                            var contradictionText = document.createElement("p");
                            contradictionText.textContent = contradiction;

                            var contradictionElementId = 'contradiction-' + date + '-' + index;
                            contradictionText.id = contradictionElementId;
                            console.log("contradiction ID", contradictionElementId);
                            contradictionsPara.appendChild(contradictionText);

                        });
                        suggestionDiv.appendChild(contradictionsPara);

                        var keepBtn = document.createElement("button");
                        keepBtn.textContent = "Keep Contradiction";
                        keepBtn.onclick = function() { handleKeepContradiction(suggestion, date + '-' + index); };
                        suggestionDiv.appendChild(keepBtn);

                        var removeBtn = document.createElement("button");
                        removeBtn.textContent = "Remove Contradiction";
                        removeBtn.onclick = function() { handleRemoveContradiction(suggestion, date + '-' + index); };
                        suggestionDiv.appendChild(removeBtn);
                        
                    }
                    dateSection.appendChild(suggestionDiv);
                    
                    var acceptBtn = document.createElement("button");
                    acceptBtn.textContent = "Accept";
                    acceptBtn.id = 'acceptBtn-' + date + '-' + index;
                    acceptBtn.onclick = function() { handleSuggestionAccept(suggestion, date + '-' + index); };  // Use date-index as identifier
                    suggestionDiv.appendChild(acceptBtn);

                    var rejectBtn = document.createElement("button");
                    rejectBtn.textContent = "Reject";
                    rejectBtn.id = 'rejectBtn-' + date + '-' + index;
                    rejectBtn.onclick = function() { handleSuggestionReject(suggestion, date + '-' + index); };  // Use date-index as identifier
                    suggestionDiv.appendChild(rejectBtn);

                });
                    // Append the dateSection to modalContent
                modalContent.appendChild(dateSection);
            });
            var acceptAllBtn = document.createElement("button");
                acceptAllBtn.textContent = "Accept All";
                acceptAllBtn.onclick = handleAcceptAll;
                modalContent.appendChild(acceptAllBtn);

                // Create and append "Reject All" button
                var rejectAllBtn = document.createElement("button");
                rejectAllBtn.textContent = "Reject All";
                rejectAllBtn.onclick = handleRejectAll;
                modalContent.appendChild(rejectAllBtn);

            // Create and append "Close" button
            var closeBtn = document.createElement("button");
            closeBtn.textContent = "Close";
            closeBtn.onclick = closeModal;
            modalContent.appendChild(closeBtn);

            document.getElementById('suggestionsModal').style.display = 'block';
        }
```
In the last part, we need functions to handle all and reject all suggestions. That is, what to do when the buttons are clicked. 
- In this case, it sends a post request to the backend with action flag as 'accept' or 'reject'. The backend will then alter the dictionary based on these flags.

```
function closeModal() {
            document.getElementById('suggestionsModal').style.display = 'none';
        }
        function handleAcceptAll() {
        fetch('/bulk_record_suggestion', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({
                suggestions: suggestions,
                action: "accept"
            })
        })
        .then(response => response.json())
        .then(data => closeModal());
    }

        function handleRejectAll() {
            fetch('/bulk_record_suggestion', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({
                    suggestions: suggestions,
                    action: "reject"
                })
            })
            .then(response => response.json())
            .then(data => closeModal());
        }

```
If the user decides to individually accept or reject suggestions - we need function to handle that request too. 
- Here if the accept or reject buttons are pressed, a post request is sent to the backend with the action flag changed. It also changes the text color dynamically to red or green.

```
function handleSuggestionAccept(suggestion, identifier) {
            console.log("Accepting with identifier:", identifier);
            // Extract the date from the suggestion object
            var parts = identifier.split('-');
            var dateFromIdentifier = parts.slice(0, 3).join('-');
            var suggestionElementId = 'suggestion-' + identifier;
            var acceptButtonId = 'acceptBtn-' + identifier;
            console.log("Target element:", acceptButtonId);

            fetch('/record_suggestion', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    suggestion: suggestion,
                    action: "accept",
                    date: dateFromIdentifier  // Use the extracted date from the suggestion
                })
            })
            .then(response => response.json())
            .then(data => {
                var acceptButton = document.getElementById(acceptButtonId);
                console.log("accept element found:", acceptButton);
                if (acceptButton) {
                    acceptButton.style.backgroundColor = 'green';
                    acceptButton.style.color = 'white'; 
                    acceptButton.disabled = true;
                    }
                });
            }


        function handleSuggestionReject(suggestion, identifier) {
            // Assuming rejection doesn't need to be sent to the server, or implement similarly to accept
            var parts = identifier.split('-');
            var dateFromIdentifier = parts.slice(0, 3).join('-');
            var rejectButtonId = 'rejectBtn-' + identifier;
            console.log("Target element:", rejectButtonId);

            fetch('/record_suggestion', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    suggestion: suggestion,  // Pass the specific suggestion object
                    action: "reject",
                    date: dateFromIdentifier  // Use the extracted date from the suggestion
                })
            })
            .then(response => response.json())
            .then(data => {
                var rejectButton = document.getElementById(rejectButtonId);
                console.log("reject element found:", rejectButton);
                if (rejectButton) {
                    rejectButton.style.backgroundColor = 'red';
                    rejectButton.style.color = 'white'; 
                    rejectButton.disabled = true;
                    }
                });
            }

```
We also need to handle contradictions. That is what to do when they are accepted or rejected.
- These functions send a post request to backend with action as 'remove' or 'keep' depending on the contradiction is accepted or rejected. 
- The text turns green if the contradiction is accepted and red otherwise.

```
function handleRemoveContradiction(suggestion, identifier) {
                console.log("Removing contradiction for identifier:", identifier);
                fetch('/record_contradiction', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        suggestion: suggestion,
                        action: "remove",
                        identifier: identifier
                    })
                })
                .then(response => response.json())
                .then(data => {
                    // Once the contradiction is successfully removed on the server, update the UI
                    var contradictionElementId = 'contradiction-' + identifier;
                    console.log("Contradiction element id:", contradictionElementId);
                    var contradictionElement = document.getElementById(contradictionElementId);
                    console.log("Contradiction element found:", contradictionElement);
                    if (contradictionElement) {
                        contradictionElement.style.color = 'red';
                        contradictionElement.style.textDecoration = 'line-through';
                        contradictionElement.disabled = true;
                    }
                })
                .catch(error => {
                    console.error('Error removing contradiction:', error);
                });
            }

            function handleKeepContradiction(suggestion, identifier) {
                console.log("Keeping contradiction for identifier:", identifier);
                fetch('/record_contradiction', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        suggestion: suggestion, 
                        action: "keep",         
                        identifier: identifier
                    })
                })
                .then(response => response.json())
                .then(data => {
                    
                    var contradictionElementId = 'contradiction-' + identifier;
                    console.log("Contradiction element id:", contradictionElementId);
                    var contradictionElement = document.getElementById(contradictionElementId);
                    console.log("Contradiction element found:", contradictionElement);
                    if (contradictionElement) {
                        contradictionElement.style.color = 'green'; 
                        contradictionElement.style.textDecoration = 'none';
                        contradictionElement.disabled = true;
                    }
                })
                .catch(error => {
                    console.error('Error keeping contradiction:', error);
                });
            }
```

## Revisiting Backend functions 🚧
We made some post calls to functions on the backend which were not defined earlier. Let's define them now. 

We need to make changes to the base dictionary when the user accepts or rejects a suggestion. In the function below we look iterate over all the facts in factsByDay dictionary and check whether facts matches the suggestion. If yes, we change the corresponding action flag. This will be used in view facts page.

```
@app.route('/record_suggestion', methods=['POST'])
def record_suggestion():
    data = request.get_json()
    suggestion = data.get('suggestion')
    action = data.get('action')
    date_key = data.get('date')
    
    if action == "accept":
        for fact in questions_and_facts["factsByDay"].get(date_key):
            if fact["text"] == suggestion['text']:
                
                fact["action"] = "accepted" 

    elif action == "reject":
        for fact in questions_and_facts["factsByDay"].get(date_key):
            if fact["text"] == suggestion['text']:                
                fact['action'] = "rejected"
        
    return jsonify({"status": "success"})
```
Now we need a function to record what contradiction is accepted and rejected (if any).
- In this we look at the action received and change the contradiction flag. We do that by looking at factsByDay dictionary for each date and index.
- The index will be defined by an identifier tag that will come with each post request.

```
@app.route('/record_contradiction', methods=['POST'])
def record_contradiction():
    data = request.get_json()
    suggestion_id = data.get('identifier')
    action = data.get('action')
    parts = suggestion_id.split('-')
    date_key = '-'.join(parts[0:3])
    index = int(parts[-1])
    
    suggestion = questions_and_facts["factsByDay"][date_key][index]
    
    if action == "keep":
        if 'contradictions' in suggestion:
            suggestion['contradictions']['status'] = 'keep'
    elif action == "remove":
        
        if 'contradictions' in suggestion:
            suggestion['contradictions']['status'] = 'remove'
    
    return jsonify({"status": "success"})
```
Finally, we will write a function to bulk record the suggestions. The logic is similar as above, but in this case we don't use an identifier. Instead we iterate over suggestion list to check the corresponding actions and make the changes in the base dictionary accordingly. 

```

@app.route('/bulk_record_suggestion', methods=['POST'])
def bulk_record_suggestion():
    data = request.get_json()
    suggestions = data.get('suggestions', {})
    action = data.get('action', '')
    
    for date_key, suggestions_list in suggestions.items():
        for suggestion in suggestions_list:
            suggestion_text = suggestion.get('text')

            if action == "accept":
                
                if date_key in questions_and_facts["factsByDay"]:
                    for fact in questions_and_facts["factsByDay"][date_key]:
                        if fact['text'] == suggestion_text:
                            fact['action'] = 'accepted'
                            
            elif action == "reject":
                
                if date_key in questions_and_facts["factsByDay"]:
                    for fact in questions_and_facts["factsByDay"][date_key]:
                        if fact['text'] == suggestion_text:
                            fact['action'] = 'rejected'

    return jsonify({"status": "success"})
```
## Final Frontend 🚩
### View facts page
Now we can define how to show the facts. At this point, the user would have already submitted the question and documents URLs - accepted/rejected the suggestions and would expect something on the facts page. 
- The function below updates the facts dynamically then clicked on a date on the timeline.


```
function updateFactsByDate(selectedDate) {
    console.log("Updating facts for date:", selectedDate);
    var factsContainer = document.getElementById('factsContainer');
    var facts = document.querySelectorAll('#factsContainer li');
    var dateMessage = document.getElementById('dateMessage');
    if (!dateMessage) {
        dateMessage = document.createElement('h3');
        dateMessage.id = 'dateMessage';
        factsContainer.insertBefore(dateMessage, factsContainer.firstChild);
    }
    dateMessage.textContent = 'All facts on ' + selectedDate + ':';

    if (selectedDate) {
        dateMessage.textContent = 'All facts on ' + selectedDate + ':';
        facts.forEach(function(fact) {
            const shouldDisplay = fact.getAttribute('data-should-display') === 'true';
            fact.style.display = fact.dataset.date === selectedDate ? '' : 'none';
        });
    } else {
        dateMessage.textContent = 'All facts:';
        facts.forEach(function(fact, index) {
            const shouldDisplay = fact.getAttribute('data-should-display') === 'true';
            fact.style.display = ''; 
            fact.innerHTML = fact.innerHTML.replace(/^\d+\.\s*/, '');
            if (!fact.innerHTML.startsWith('(Date:')) {
                fact.innerHTML = '(' + fact.dataset.date + ') ' + fact.innerHTML;
            }
        });
    }
}
```
Additionally, a timeline is initiated in Vis.js to make the UI more user friendly. The logic is as follows:
- DOM Initialization: Waits for the DOM to load.
- Data Preparation: Loops through facts and adds them to the timeline dataset.
- Timeline Setup: Configures and creates the timeline.
- Event Handling: Adds a listener to update displayed facts when an item is selected.
- Display Update: Function to update fact display based on the selected date.
- Reset Functionality: Allows resetting the data via a POST request.

### That's a wrap! 
Check out the [github repo](https://github.com/Anushk97/Cleric_assessment/tree/main)

Check out the [deployed app](https://llm-bot-pficsoahpa-uc.a.run.app/)