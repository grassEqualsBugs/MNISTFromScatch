import { useEffect, useState } from "react";
import "./App.css";
import UserInput from "./components/UserInput";

function App() {
    /*
    useEffect(() => {
        fetch("/data", {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify({ data: "Some data" }),
        })
            .then((res) => res.json())
            .then((data) => {
                console.log("Response from /data:", data);
            })
            .catch((error) => {
                console.error("Error sending data:", error);
            });
    }, []);
    */

    const defaultGrid = Array.from({ length: 28 }, () => Array(28).fill(false));
    const [grid, setGrid] = useState(defaultGrid);

    return (
        <div className="App">
            <UserInput size="728" grid={grid} setGrid={setGrid} />
        </div>
    );
}

export default App;
