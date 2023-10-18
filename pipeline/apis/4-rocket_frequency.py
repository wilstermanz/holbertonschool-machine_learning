#!/usr/bin/env python3
"""Task 4"""

if __name__ == '__main__':
    from requests import get

    API = 'https://api.spacexdata.com/v4/'

    rocket_names = {}   # (rocket ID: rocket name)
    rockets_count = {}  # {rocket name: count}

    launches = get(API + 'launches/').json()
    for launch in launches:
        rocket_id = launch['rocket']

        # get rocket name from dictionary, else call API
        try:
            name = rocket_names[rocket_id]  # get name from dict
            rockets_count[name] += 1        # add to count

        except KeyError:
            # get name from API
            name = get(API + 'rockets/' + rocket_id).json()['name']

            # add name to dict
            rocket_names[rocket_id] = name
            rockets_count[name] = 1         # create dict entry and set to 1

    # sort alphabetically into list of tuples
    rockets_count = sorted(rockets_count.items(), key=lambda d: d[0])

    # sort list by value and print
    for rocket in sorted(rockets_count, key=lambda d: d[1], reverse=True):
        print('{}: {}'.format(rocket[0], rocket[1]))
