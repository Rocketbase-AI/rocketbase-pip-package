import itertools
import requests

class RocketAPI:
    def __init__(self):
        self.models = []
        self.selected_model = {}
        self.models_api_url = "https://europe-west1-rockethub.cloudfunctions.net/getAvailableModels"

    def get_rocket_info(self, rocket: str):
        """ Parse the Rocket identifier.

        The Rocket identifier is a String following this pattern: rocket_author/rocket_name/rocket_version
        The rocket_version is not mandatory and if not provided by the user the newest version will be returned.

        Args:
            rocket (str): Rocket identifier that needs to be parsed
        """
        assert len(rocket) > 0, 'Please specify the rocket you want to get.'
        
        # Parse the Rocket url
        rocket_parsed = rocket.split('/')
        assert len(rocket_parsed) > 1, 'Please provide more information about the rocket you want to get.'

        rocket_author  = rocket_parsed[0]
        rocket_name    = rocket_parsed[1]

        print('Looking for the Rocket ' + rocket_name + ' made by ' + rocket_author + '...')
        payload = {'author': rocket_author, 'model': rocket_name}
        rocket_version = ''
        if len(rocket_parsed) > 2:
            rocket_version = rocket_parsed[2]
            payload['version'] = rocket_version
        res = requests.get(self.models_api_url, params=payload)
        
        # if status != 200 then database is broken
        assert res.status_code == 200, 'Database error. Please try again later.'
        self.models = res.json()

        # Test that the rocket exists
        assert len(self.models) > 0, rocket + 'rocket cannot be found from our database. Please check the spelling.'
        print('{models_len} model versions found from the database.'.format(models_len=len(self.models)))
        
        # TODO: Select using some better logic
        self.selected_model = self.models[0]
        if rocket_version:
            print('Version ' + rocket_version + 'selected.')
        else:
            print('You didn\'t specify the version so the newest one is used.')

        return rocket_author, rocket_name, rocket_version

    def get_rocket_url(self, rocket_author: str, rocket_name: str, rocket_version: str):
        """ Get the url from which to download the Rocket.

        Args:
            rocket_author (str): Username of the author of the Rocket
            rocket_name (str): Name of the rocket 
            rocket_version (str): Version of the Rocket
        """
        return self.selected_model['modelFilePath']
    
    def get_rocket_folder(self, rocket_author: str, rocket_name: str, rocket_version: str):
        """ Get the name of the folder where the Rocket is unpacked.

        Args:
            rocket_author (str): Username of the author of the Rocket
            rocket_name (str): Name of the rocket 
            rocket_version (str): Version of the Rocket
        """
        return self.selected_model['folderName']
    
    def get_rocket_last_version(self, rocket_author: str, rocket_name: str):
        """Get the last version of a Rocket.

        Args:
            rocket_author (str): Username of the author of the Rocket
            rocket_name (str): Name of the rocket 
        """
        # Verify the rocket exist
        assert rocket_author in self.hangar.keys(), rocket_author + ' can\'t be found as an author.'
        assert rocket_name in self.hangar[rocket_author].keys(), rocket_name + ' can\'t be found as a rocket from ' + rocket_author

        # Get list of versions for a specific Rockets
        list_versions = [v[1:] for v in self.hangar[rocket_author][rocket_name].keys() if v.startswith('v')]

        mainVersion = 0
        minorVersion = 'a'

        for version in list_versions:
            v = ["".join(x) for _, x in itertools.groupby(version, key=str.isdigit)]
            temp_mainVersion = int(v[0])
            temp_minorVersion = v[1]

            assert len(temp_minorVersion) == 1, 'Automatic selection of the newest version doesn\'t support minor version made of more than 1 char.' 

            if temp_mainVersion == mainVersion:
                if temp_minorVersion > minorVersion:
                    minorVersion = temp_minorVersion
            elif temp_mainVersion > mainVersion:
                mainVersion = temp_mainVersion
                minorVersion = 'a'
                if temp_minorVersion > minorVersion:
                    minorVersion = temp_minorVersion
        
        return 'v' + str(mainVersion) +  minorVersion
