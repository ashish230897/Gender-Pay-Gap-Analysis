function predict(){

  let age = document.getElementById("age").value;
  let gender = document.getElementById("genders").value;
  let education = document.getElementById("education").value;
  let country = document.getElementById("country").value;
  let major = document.getElementById("major").value;
  let profession = document.getElementById("profession").value;
  let industry = document.getElementById("industry").value;
  let experience = document.getElementById("experience").value;
 
  
  let data = {"Gender": gender, "Age": age, "Country": country, "Education": education, "Major": major,
              "Profession": profession, "Industry": industry, "Experience": experience};
  console.log(data);

  fetch("http://127.0.0.1:12345/classify", {
  method: "POST",
  mode: 'cors',
  headers: {'Content-Type': 'application/json'}, 
  body: JSON.stringify(data)
  // }).then(res => console.log(res.json()));
  }).then(res => res.json()).then(
    (myBlob) => {
      console.log("result is ", myBlob["class"]);
      document.getElementById("result").innerHTML = "According to your details, your salary range should lie in the range: " 
      + myBlob["class"] + " dollars";

    }
  );
}