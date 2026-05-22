const express = require("express");
const cors = require("cors");

const faceRoutes = require("./routes/faceRoutes");

const app = express();

app.use(cors());
app.use(express.json());

app.use("/api", faceRoutes);

app.get("/", (req, res) => {
  res.send("Backend FaceFit Barber jalan");
});

const PORT = 3000;

app.listen(PORT, () => {
  console.log(`Server running on port ${PORT}`);
});
