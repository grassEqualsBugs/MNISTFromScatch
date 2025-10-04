import { useState, useEffect, useCallback } from "react";
import "./App.css";
import UserInput from "./components/UserInput";

function App() {
    const defaultGrid = Array.from({ length: 28 }, () => Array(28).fill(0));
    const [grid, setGrid] = useState(defaultGrid);
    const [prediction, setPrediction] = useState([]);

    const onPredictPressed = useCallback(() => {
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
    }, [grid]);

    const onResetPressed = useCallback(() => {
        setGrid(defaultGrid);
        setPrediction([]);
    }, [defaultGrid]);

    // effect for handling global key presses
    useEffect(() => {
        const handleKeyDown = (event) => {
            if (event.key === "r" || event.key === "R") {
                onResetPressed();
            }
            if (event.key === "p" || event.key === "P") {
                onPredictPressed();
            }
        };

        window.addEventListener("keydown", handleKeyDown);

        return () => {
            window.removeEventListener("keydown", handleKeyDown);
        };
    }, [onResetPressed, onPredictPressed]);

    return (
        <div className="App">
            <UserInput size="728" grid={grid} setGrid={setGrid} />
            <div className="prediction">
                <button onClick={onResetPressed}>RESET (R)</button>
                <button onClick={onPredictPressed}>PREDICT (P)</button>
                <p className="predictionText">Prediction:</p>
                <ul>
                    {prediction.map(({ key, percent }, n) => (
                        <li key={key}>
                            {n}: {(percent * 100).toFixed(3)}%
                        </li>
                    ))}
                </ul>
            </div>
        </div>
    );
}

export default App;
