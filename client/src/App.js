import { useState } from "react";

function App() {
    const [text, setText] = useState("");
    const [response, setResponse] = useState("");

    const handleTextChange = (event) => {
        setText(event.target.value);
    };

    const handleSubmit = (event) => {
        event.preventDefault();
        fetch("/data", {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify({ text }),
        })
            .then((res) => res.json())
            .then((data) => {
                console.log("Response from /data:", data);
                setResponse(JSON.stringify(data));
            })
            .catch((error) => {
                console.error("Error sending data:", error);
            });
    };

    return (
        <div>
            <h1>Send data to Flask</h1>
            <form onSubmit={handleSubmit}>
                <input type="text" value={text} onChange={handleTextChange} />
                <button type="submit">Send</button>
            </form>
            <h2>Response from server:</h2>
            <p>{response}</p>
        </div>
    );
}

export default App;
