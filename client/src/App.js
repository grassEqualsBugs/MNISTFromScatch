import { useState } from "react";
import "./App.css";
import UserInput from "./components/UserInput";

function App() {
    const defaultGrid = Array.from({ length: 28 }, () => Array(28).fill(0));
    const [grid, setGrid] = useState(defaultGrid);
    const [prediction, setPrediction] = useState([]);

    function onPredictPressed() {
        fetch("/predict", {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify({ activations: grid }),
        })
            .then((res) => res.json())
            .then((data) => {
                console.log("Response from /predict:", data);
                setPrediction(
                    data.prediction.map((percent, i) => {
                        return { key: i, percent: percent };
                    }),
                );
            })
            .catch((error) => {
                console.error("Error sending data:", error);
            });
    }

    return (
        <div className="App">
            <UserInput size="728" grid={grid} setGrid={setGrid} />
            <div className="prediction">
                <p>Press R to reset</p>
                <button onClick={onPredictPressed}>PREDICT</button>
                <p className="predictionText">Prediction:</p>
                <ul>
                    {prediction.map(({ key, percent }, n) => (
                        <li>
                            {n}: {(percent * 100).toFixed(3)}%
                        </li>
                    ))}
                </ul>
            </div>
        </div>
    );
}

export default App;
