let feature_descriptions = new Map();
feature_descriptions.set("Unemployment", "The unemployed include all persons who are temporarily not in employment or who work less than 15 hours per week and are looking for employment subject to social insurance contributions of at least 15 hours per week. They must reside in the Federal Republic of Germany, be at least 15 years of age, and not yet have reached the retirement age. In addition, they must have registered as unemployed in person at an employment agency or job center. <br /><br /> Unit: % of population <br /><br /> © Statistisches Bundesamt (Destatis), 2025");
feature_descriptions.set("Recipients of Benefits", "These statistics collect data on beneficiaries who received at least one of the following types of assistance at some point during the reporting year: health assistance, integration assistance for disabled persons, nursing care assistance, assistance in overcoming particular social difficulties, assistance in other life situations. <br /><br /> Unit: Recipients count per 1000 inhabitants <br /><br /> © Statistisches Bundesamt (Destatis), 2025");
feature_descriptions.set("Expenditure on Public Schools", "Expenditure on public schools per pupil <br /><br /> Unit: EUR per pupil <br /><br /> © Statistisches Bundesamt (Destatis), 2025");
feature_descriptions.set("Graduation Rates", "Percentage of university graduates (first degree only) in the population of the corresponding age. Quotas are calculated for individual birth cohorts and then added together. <br /><br /> Unit: % graduation rate <br /><br /> © Statistisches Bundesamt (Destatis), 2025");


function onLoad(){
    var labels = document.getElementsByTagName("label");
    var infoBox = document.getElementById("info-box");

    for (let i = 0; i < labels.length; i++) {
        labels[i].addEventListener('mouseover', function () {
            var labelText = labels[i].textContent;
            var description = feature_descriptions.get(labelText);
            if (description != undefined ){
                infoBox.innerHTML = description;
            }
        });
    }
}

var millisecondsToWait = 5000;
setTimeout(onLoad, millisecondsToWait);

