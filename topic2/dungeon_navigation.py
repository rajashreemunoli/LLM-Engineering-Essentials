import random
from typing import Dict, List, Set, Tuple, Optional, Any
import copy

import matplotlib.pyplot as plt
import networkx as nx

class Key:
    def __init__(self, id: int, description: str):
        """
        Initialize a key with a id and description.
        
        Args:
            id: Unique identifier for the key
            description: Text description of the key
        """
        self.id = id
        self.description = description
    
    def __repr__(self) -> str:
        return f"Key({self.id}, '{self.description}')"


class Door:
    def __init__(self, id: int, description: str, 
                 connects: Tuple[int, int], locked: bool = False, 
                 key_id: Optional[int] = None):
        """
        Initialize a door with properties and connection information.
        
        Args:
            id: Unique identifier for the door
            description: Text description of the door
            connects: Tuple of room ids that this door connects
            locked: Whether the door is initially locked
            key_id: The key id that can unlock this door (if any)
        """
        self.id = id
        self.description = description
        self.connects = connects  # (room1_id, room2_id)
        self.locked = locked
        self.key_id = key_id  # Which key unlocks this door (if any)
    
    def can_be_unlocked_by(self, key_id: int) -> bool:
        """Check if this door can be unlocked by a specific key."""
        return self.key_id == key_id
    
    def get_other_room(self, current_room: int) -> int:
        """Get the room on the other side of the door."""
        if current_room == self.connects[0]:
            return self.connects[1]
        elif current_room == self.connects[1]:
            return self.connects[0]
        else:
            raise ValueError(f"Room {current_room} is not connected to this door")
    
    def __repr__(self) -> str:
        lock_status = "locked" if self.locked else "unlocked"
        return f"Door({self.id}, '{self.description}', {self.connects}, {lock_status})"


class Room:
    def __init__(self, id: int, description: str):
        """
        Initialize a room with a id and description.
        
        Args:
            id: Unique identifier for the room
            description: Text description of the room
        """
        self.id = id
        self.description = description
        self.doors: List[int] = []  # Door ids connected to this room
        self.keys: List[int] = []  # Key ids initially in this room
    
    def add_door(self, door_id: int) -> None:
        """Add a door connection to this room."""
        if door_id not in self.doors:
            self.doors.append(door_id)
    
    def add_key(self, key_id: int) -> None:
        """Add a key to this room."""
        if key_id not in self.keys:
            self.keys.append(key_id)
    
    def remove_key(self, key_id: int) -> None:
        """Remove a key from this room (when picked up)."""
        if key_id in self.keys:
            self.keys.remove(key_id)
    
    def __repr__(self) -> str:
        return f"Room({self.id}, '{self.description}', doors={self.doors}, keys={self.keys})"


class AgentState:
    def __init__(self, start_room: int):
        """
        Initialize the agent's state within the environment.
        
        Args:
            start_room: The room id where the agent starts
        """
        self.current_room = start_room
        self.keys_possessed: List[int] = []  # Key ids the agent has
        self.rooms_explored: Set[int] = {start_room}  # Room ids the agent has seen
        self.doors_seen: Set[int] = set()  # Door ids the agent has seen
    
    def add_key(self, key_id: int) -> None:
        """Add a key to the agent's inventory."""
        if key_id not in self.keys_possessed:
            self.keys_possessed.append(key_id)
    
    def has_key(self, key_id: int) -> bool:
        """Check if the agent has a specific key."""
        return key_id in self.keys_possessed
    
    def explore_room(self, room_id: int) -> None:
        """Mark a room as explored."""
        self.rooms_explored.add(room_id)
    
    def see_door(self, door_id: int) -> None:
        """Mark a door as seen."""
        self.doors_seen.add(door_id)
    
    def move_to(self, room_id: int) -> None:
        """Move the agent to a new room."""
        self.current_room = room_id
        self.explore_room(room_id)
    
    def __repr__(self) -> str:
        return (f"AgentState(room={self.current_room}, "
                f"keys={self.keys_possessed}, "
                f"explored={self.rooms_explored})")


class DungeonNavigationEnvironment:
    def __init__(self, name: str = "Default Environment"):
        """
        Initialize the adventure environment.
        
        Args:
            name: Name of this environment configuration
        """
        self.name = name
        self.rooms: Dict[int, Room] = {}
        self.doors: Dict[int, Door] = {}
        self.keys: Dict[int, Key] = {}
        self.start_room: Optional[int] = None
        self.target_room: Optional[int] = None
        self.agent_states: Dict[str, AgentState] = {}  # Multiple agents can exist
    
    def add_room(self, room: Room) -> None:
        """Add a room to the environment."""
        self.rooms[room.id] = room
    
    def add_door(self, door: Door) -> None:
        """
        Add a door to the environment and connect the rooms.
        
        Args:
            door: The door to add
        """
        self.doors[door.id] = door
        # Add the door to both connected rooms
        room1, room2 = door.connects
        self.rooms[room1].add_door(door.id)
        self.rooms[room2].add_door(door.id)
    
    def add_key(self, key: Key, room_id: int) -> None:
        """
        Add a key to the environment and place it in a room.
        
        Args:
            key: The key to add
            room_id: Room where the key is initially located
        """
        self.keys[key.id] = key
        self.rooms[room_id].add_key(key.id)
    
    def set_start_room(self, room_id: int) -> None:
        """Set the starting room for agents."""
        if room_id in self.rooms:
            self.start_room = room_id
        else:
            raise ValueError(f"Room {room_id} does not exist")
    
    def set_target_room(self, room_id: int) -> None:
        """Set the target room (goal room) for agents."""
        if room_id in self.rooms:
            self.target_room = room_id
        else:
            raise ValueError(f"Room {room_id} does not exist")
    
    def initialize_agent(self, agent_id: str) -> None:
        """
        Initialize a new agent in the environment.
        
        Args:
            agent_id: Unique identifier for the agent
        
        Returns:
            Initial observation of the environment
        """
        if self.start_room is None:
            raise ValueError("Start room must be set before initializing an agent")
        
        self.agent_states[agent_id] = AgentState(self.start_room)
        
        # Return initial observation (description of start room and visible doors)
        return self._get_room_observation(agent_id)
    
    def _get_room_observation(self, agent_id: str) -> Dict[str, Any]:
        """
        Get the observation for the agent's current room.
        
        Args:
            agent_id: The agent requesting the observation
        
        Returns:
            Dictionary with room description and visible door information
        """
        agent_state = self.agent_states[agent_id]
        current_room = self.rooms[agent_state.current_room]
        
        # Get information about doors in this room
        doors_info = []
        for door_id in current_room.doors:
            door = self.doors[door_id]
            agent_state.see_door(door_id)
            doors_info.append({
                "id": door.id,
                "description": door.description,
                "locked": door.locked
            })
        
        return {
            "room_id": current_room.id,
            "room_description": current_room.description,
            "doors": doors_info
        }
    
    def move(self, agent_id: str, door_id: int) -> Dict[str, Any]:
        """
        Attempt to move the agent through a door.
        
        Args:
            agent_id: The agent attempting to move
            door_id: The door to move through
        
        Returns:
            Result of the movement attempt, with new room observation if successful
        """
        try:
            door_id = int(door_id)
        except:
            return {"success": False, "message": "The door id is wrong. Use the correct door id"}
        
        agent_state = self.agent_states[agent_id]
        current_room = self.rooms[agent_state.current_room]
        
        # Check if the door exists in the current room
        if door_id not in current_room.doors:
            return {"success": False, "message": "There is no such door in this room."}
        
        door = self.doors[door_id]
        
        # Check if the door is locked
        if door.locked:
            return {"success": False, "message": "The door is locked. You need a key to open it."}
        
        # Move to the other room
        destination_room = door.get_other_room(agent_state.current_room)
        agent_state.move_to(destination_room)
        
        # Check if this is the target room
        if destination_room == self.target_room:
            return {
                "success": True,
                "victory": True,
                "message": "Congratulations! You have reached the room with the priceless artifact and completed your mission!",
                **self._get_room_observation(agent_id)
            }
        
        # Return observation of the new room
        return {
            "success": True,
            "victory": False,
            "message": "You moved through the door successfully.",
            **self._get_room_observation(agent_id)
        }
    
    def search_for_keys(self, agent_id: str) -> Dict[str, Any]:
        """
        Search the current room for keys.
        
        Args:
            agent_id: The agent searching for keys
        
        Returns:
            Keys found in the room
        """
        agent_state = self.agent_states[agent_id]
        current_room = self.rooms[agent_state.current_room]
        
        keys_found = []
        for key_id in current_room.keys:
            key = self.keys[key_id]
            keys_found.append({
                "id": key.id,
                "description": key.description
            })
            agent_state.add_key(key_id)
            current_room.remove_key(key_id)
        
        if keys_found:
            return {
                "success": True,
                "message": f"You found {len(keys_found)} key(s)!",
                "keys": keys_found
            }
        else:
            return {
                "success": False,
                "message": "You searched the room but found no keys.",
                "keys": []
            }
    
    def try_unlock(self, agent_id: str, door_id: int, key_id: int) -> Dict[str, bool]:
        """
        Try to unlock a door with a specific key.
        
        Args:
            agent_id: The agent attempting to unlock
            door_id: The door to unlock
            key_id: The key to use
        
        Returns:
            Result of the unlock attempt
        """
        try:
            door_id = int(door_id)
        except:
            return {"success": False, "message": "The door id is wrong. Use the correct door id"}
        
        try:
            key_id = int(key_id)
        except:
            return {"success": False, "message": "The key id is wrong. Use the correct key id"}

        agent_state = self.agent_states[agent_id]
        
        # Check if the agent has the key
        if not agent_state.has_key(key_id):
            return {
                "success": False,
                "message": "You don't have that key."
            }
        
        # Check if the door exists and is visible to the agent
        if door_id not in agent_state.doors_seen:
            return {
                "success": False,
                "message": "You haven't seen that door."
            }
        
        door = self.doors[door_id]
        
        # Check if the door is already unlocked
        if not door.locked:
            return {
                "success": False,
                "message": "The door is already unlocked."
            }
        
        # Try to unlock the door
        if door.can_be_unlocked_by(key_id):
            door.locked = False
            return {
                "success": True,
                "message": "The key worked! The door is now unlocked."
            }
        else:
            return {
                "success": False,
                "message": "The key doesn't fit this lock."
            }
    
    def get_agent_state(self, agent_id: str) -> Dict[str, Any]:
        """
        Get the current state of an agent (for debugging or display).
        
        Args:
            agent_id: The agent to query
        
        Returns:
            Current state information for the agent
        """
        if agent_id not in self.agent_states:
            return {"error": "Agent does not exist"}
        
        agent_state = self.agent_states[agent_id]
        
        keys_info = []
        for key_id in agent_state.keys_possessed:
            key = self.keys[key_id]
            keys_info.append({
                "id": key.id,
                "description": key.description
            })
        
        return {
            "current_room": agent_state.current_room,
            "rooms_explored": list(agent_state.rooms_explored),
            "doors_seen": list(agent_state.doors_seen),
            "keys": keys_info
        }
    
    def reset_agent(self, agent_id: str) -> Dict[str, Any]:
        """
        Reset an agent to the starting state.
        
        Args:
            agent_id: The agent to reset
        
        Returns:
            Initial observation of the environment
        """
        return self.initialize_agent(agent_id)
    
    def clone(self) -> 'DungeonNavigationEnvironment':
        """Create a deep copy of the environment."""
        return copy.deepcopy(self)


class EnvironmentFactory:
    """Factory class to create different environment configurations."""
    
    @staticmethod
    def create_simple_environment() -> DungeonNavigationEnvironment:
        """
        Create a simple environment with 3 rooms, 2 doors (1 locked), and 1 key.
        
        Returns:
            A simple adventure environment
        """
        env = DungeonNavigationEnvironment("Simple Dungeon")
        
        # Create rooms
        starting_room = Room(1, "A small, dimly lit chamber with stone walls. Dust motes dance in the air, and cobwebs adorn the corners.")
        middle_room = Room(2, "A circular room with a high domed ceiling. Ancient symbols are carved into the floor, glowing with a faint blue light.")
        treasure_room = Room(3, "A vast chamber with a vaulted ceiling, its ancient stone arches entwined with glowing bioluminescent vines. At the center, on a pedestal, sits the priceless artifact you seek.")
        
        # Add rooms to environment
        env.add_room(starting_room)
        env.add_room(middle_room)
        env.add_room(treasure_room)
        
        # Create doors
        door1 = Door(101, "A wooden door with iron bands. It looks sturdy but not locked.", (1, 2))
        door2 = Door(102, "A heavy stone door with intricate carvings of serpents. A keyhole shaped like a crescent moon is visible.", (2, 3), locked=True, key_id=201)
        
        # Add doors to environment
        env.add_door(door1)
        env.add_door(door2)
        
        # Create key and place it in the starting room
        key = Key(201, "A silver key with a crescent moon-shaped head, glinting in the light.")
        env.add_key(key, 1)
        
        # Set start and target rooms
        env.set_start_room(1)
        env.set_target_room(3)
        
        return env
    
    @staticmethod
    def create_medium_environment() -> DungeonNavigationEnvironment:
        """
        Create a medium complexity environment with 5 rooms, 6 doors (3 locked), and 4 keys (including a red herring).
        
        Returns:
            A medium complexity adventure environment
        """
        env = DungeonNavigationEnvironment("Medium Dungeon")
        
        # Create rooms
        entrance = Room(1, "A grand entrance hall with faded tapestries and a dusty chandelier hanging from the ceiling.")
        east_wing = Room(2, "A library with towering bookshelves. Most books have rotted away, but a few ancient tomes remain intact.")
        west_wing = Room(3, "An old kitchen with rusted utensils and a large stone oven. The smell of ancient herbs still lingers.")
        basement = Room(4, "A damp cellar with wine racks and storage barrels. Water drips from the ceiling, creating small puddles on the floor.")
        vault = Room(5, "A secure vault with walls of solid stone. In the center, bathed in a shaft of light from above, rests the priceless artifact you seek.")
        
        # Add rooms to environment
        env.add_room(entrance)
        env.add_room(east_wing)
        env.add_room(west_wing)
        env.add_room(basement)
        env.add_room(vault)
        
        # Create doors
        door1 = Door(101, "A door to the east with a carved owl above the frame.", (1, 2))
        door2 = Door(102, "A door to the west with a carved bear above the frame.", (1, 3))
        door3 = Door(103, "A trapdoor in the floor, secured with a rusted padlock.", (1, 4), locked=True, key_id=201)
        door4 = Door(104, "A small door hidden behind a bookshelf, requiring a triangular key.", (2, 5), locked=True, key_id=203)
        door5 = Door(105, "A heavy metal door with a circular lock mechanism.", (3, 4), locked=True, key_id=202)
        door6 = Door(106, "A secret passage behind a loose stone in the wall.", (4, 5))
        
        # Add doors to environment
        env.add_door(door1)
        env.add_door(door2)
        env.add_door(door3)
        env.add_door(door4)
        env.add_door(door5)
        env.add_door(door6)
        
        # Create keys and place them
        rusty_key = Key(201, "A rusty iron key, possibly for a padlock.")
        circular_key = Key(202, "A circular key with gear-like teeth.")
        triangular_key = Key(203, "A strange triangular key made of brass.")
        red_herring = Key(204, "A golden key with intricate engravings, seems valuable but doesn't fit any lock you've seen.")
        
        env.add_key(rusty_key, 3)  # Kitchen
        env.add_key(circular_key, 2)  # Library
        env.add_key(triangular_key, 4)  # Basement
        env.add_key(red_herring, 1)  # Entrance (red herring key)
        
        # Set start and target rooms
        env.set_start_room(1)
        env.set_target_room(5)
        
        return env
    
    @staticmethod
    def create_complex_environment() -> DungeonNavigationEnvironment:
        """
        Create a complex environment with 8 rooms, 12 doors (6 locked), and multiple keys including red herrings.
        
        Returns:
            A complex adventure environment
        """
        env = DungeonNavigationEnvironment("Complex Dungeon")
        
        # Create rooms
        entrance = Room(1, "The entrance chamber to an ancient temple. Huge stone pillars support the ceiling, carved with images of forgotten deities.")
        ritual_room = Room(2, "A circular chamber with an altar at its center. Dried blood stains the stone, and ritual implements lie scattered about.")
        meditation = Room(3, "A peaceful meditation chamber with cushions arranged in a circle. Faint incense still scents the air.")
        treasury = Room(4, "A room once filled with riches. Now mostly empty, but golden glints can still be seen in the corners.")
        guardian = Room(5, "A large hall with statues of warriors lining the walls. Their stone eyes seem to follow your movements.")
        puzzle_room = Room(6, "A chamber filled with strange mechanisms and puzzles. Some appear to have been solved, others remain mysterious.")
        crypt = Room(7, "A dark crypt with stone sarcophagi arranged in neat rows. The air is cold and still.")
        artifact_chamber = Room(8, "A spectacular domed chamber with a beam of light shining down on a central pedestal. The priceless artifact glows with an inner light.")
        
        # Add rooms
        for room in [entrance, ritual_room, meditation, treasury, guardian, puzzle_room, crypt, artifact_chamber]:
            env.add_room(room)
        
        # Create doors
        doors = [
            Door(101, "A massive stone door with carvings of the sun and moon.", (1, 2)),
            Door(102, "A wooden door with iron reinforcements, painted red.", (1, 3)),
            Door(103, "A golden door that gleams in the light.", (1, 4), locked=True, key_id=201),
            Door(104, "A door made of some strange metal that seems to shift colors.", (2, 5)),
            Door(105, "A simple wooden door with a silver handle.", (2, 6), locked=True, key_id=204),
            Door(107, "A door with a complex mechanical lock.", (3, 7)),
            Door(108, "A door with the image of a roaring lion.", (4, 6)),
            Door(109, "A small hidden door behind a tapestry.", (5, 7), locked=True, key_id=205),
            Door(110, "A door sealed with magical runes.", (5, 8), locked=True, key_id=203),
            Door(111, "A door that seems to be made of solidified shadows.", (6, 7)),
            Door(112, "A door of pure white marble with veins of gold.", (7, 8), locked=True, key_id=206)
        ]
        
        # Add doors
        for door in doors:
            env.add_door(door)
        
        # Create keys
        keys = [
            Key(201, "A golden key with sun symbols."),
            Key(202, "A crystalline key that refracts light."),
            Key(203, "A key inscribed with magical runes."),
            Key(204, "A silver key with a mechanical design."),
            Key(205, "A tiny key hidden inside a hollow book."),
            Key(206, "A white marble key with gold inlay."),
            Key(207, "An ornate key that seems valuable but fits no lock you've seen."),
            Key(208, "A strange key made of an unknown material, possibly just decorative.")
        ]
        
        # Place keys in rooms
        key_placements = {
            1: [207],  # Red herring in entrance
            2: [201],  # Golden key in ritual room
            3: [208],  # Another red herring in meditation room
            4: [204],  # Silver key in treasury
            5: [206],  # Marble key in guardian room
            6: [202],  # Crystal key in puzzle room
            7: [203, 205]  # Rune key and book key in crypt
        }
        
        for room_num, key_nums in key_placements.items():
            for key_num in key_nums:
                for key in keys:
                    if key.id == key_num:
                        env.add_key(key, room_num)
        
        # Set start and target rooms
        env.set_start_room(1)
        env.set_target_room(8)
        
        return env
    
    @staticmethod
    def create_custom_environment(
        num_rooms: int = 5,
        num_doors: int = 8,
        lock_percentage: float = 0.4,
        red_herring_keys: int = 1
    ) -> DungeonNavigationEnvironment:
        """
        Create a customized randomly generated environment.
        
        Args:
            num_rooms: Number of rooms to generate
            num_doors: Number of doors to generate
            lock_percentage: Percentage of doors that should be locked
            red_herring_keys: Number of keys that don't open any doors
        
        Returns:
            A randomly generated environment
        """
        env = DungeonNavigationEnvironment(f"Custom Environment ({num_rooms} rooms)")
        
        # Room descriptions for random generation
        room_descriptions = [
            "A dimly lit chamber with stone walls covered in ancient runes.",
            "A vast hall with enormous pillars reaching up to a ceiling lost in darkness.",
            "A small room with shelves filled with dusty scrolls and tomes.",
            "A chamber with a bubbling fountain at its center, the water glowing with an eerie blue light.",
            "A room dominated by a large stone table, covered with maps and strange instruments.",
            "A serene meditation chamber with cushions arranged in a circle.",
            "A dusty library with cobwebs connecting the tall bookshelves.",
            "A workshop filled with alchemical apparatus and strange ingredients.",
            "A treasure vault with empty pedestals and broken chests.",
            "A high-ceilinged chamber with colorful mosaics depicting ancient battles.",
            "A dark room with mysterious symbols etched into the floor, forming a perfect circle.",
            "A room filled with statues of warriors frozen in different combat poses.",
            "A garden chamber with exotic plants growing despite the lack of sunlight.",
            "A room housing a large astronomical model of the cosmos, slowly rotating.",
            "A chamber with walls lined with masks representing different emotions."
        ]
        
        # Door descriptions for random generation
        door_descriptions = [
            "A heavy oak door reinforced with iron bands.",
            "A door made of some strange metal that seems to absorb light.",
            "A simple wooden door with intricate carvings of forest scenes.",
            "A stone door that appears to be part of the wall until examined closely.",
            "A door made of interlocking metal plates that slide apart when unlocked.",
            "A crystalline door that distorts vision when looking through it.",
            "A door covered in mysterious runes that glow faintly in the dark.",
            "An ornate door with gold filigree depicting mythical creatures.",
            "A door that seems to be made of solidified smoke, swirling slightly.",
            "A door made from the wood of an enormous ancient tree, with the grain forming faces.",
            "A door of burnished bronze with a sun emblem at its center.",
            "A door that appears to be made of ice but isn't cold to the touch.",
            "A door woven from thousands of tiny metallic threads.",
            "A stone door with a relief carving of a sleeping dragon.",
            "A door made of obsidian, polished to a mirror finish."
        ]
        
        # Key descriptions for random generation
        key_descriptions = [
            "A silver key with an intricate head shaped like a tree.",
            "A heavy iron key with teeth that resemble mountain peaks.",
            "A delicate gold key with a head shaped like a star.",
            "A bronze key with strange symbols engraved along its length.",
            "A crystal key that catches and refracts light in rainbow patterns.",
            "A bone key, yellowed with age but still strong.",
            "A wooden key, somehow hardened to be as durable as metal.",
            "A key made of intertwined metals, copper and silver twisted together.",
            "A key that seems to be made of solidified moonlight.",
            "A key with a head shaped like a roaring lion.",
            "A key that feels unusually warm to the touch.",
            "A key that subtly changes shape when not being observed directly.",
            "A key with a head formed like a perfect circle with no markings.",
            "A key that appears to be made of some unknown greenish metal.",
            "A key with a head shaped like a six-pointed star."
        ]
        
        # Create rooms
        for i in range(1, num_rooms + 1):
            description = random.choice(room_descriptions)
            room_descriptions.remove(description)  # Ensure unique descriptions
            if not room_descriptions:  # Replenish if exhausted
                room_descriptions = list(room_descriptions)
            
            room = Room(i, description)
            env.add_room(room)
        
        # Create a connected graph to ensure all rooms are reachable
        # First, create a minimal spanning tree (MST) to ensure connectivity
        
        # Track which rooms are connected to the MST
        connected_rooms = {1}  # Start with room 1 connected
        remaining_rooms = set(range(2, num_rooms + 1))
        
        # Track which doors are part of the MST
        mst_doors = []  # These doors form the spanning tree
        
        # Create the MST
        door_id = 101
        while remaining_rooms:
            room_from = random.choice(list(connected_rooms))
            room_to = random.choice(list(remaining_rooms))
            
            description = random.choice(door_descriptions)
            door_descriptions.remove(description)
            if not door_descriptions:
                door_descriptions = list(door_descriptions)
            
            # Create a door connecting these rooms (initially unlocked)
            door = Door(door_id, description, (room_from, room_to))
            env.add_door(door)
            mst_doors.append(door_id)
            door_id += 1
            
            # Move the connected room from remaining to connected
            connected_rooms.add(room_to)
            remaining_rooms.remove(room_to)
        
        # Add some additional doors to create loops in the graph
        additional_doors = num_doors - (num_rooms - 1)  # We already added n-1 doors
        non_mst_doors = []  # Doors not in the MST (can be locked without creating dead ends)
        
        for _ in range(min(additional_doors, num_rooms * (num_rooms - 1) // 2 - (num_rooms - 1))):
            # Pick two random connected rooms that aren't already connected
            possible_connections = []
            for r1 in range(1, num_rooms + 1):
                for r2 in range(r1 + 1, num_rooms + 1):
                    # Check if these rooms already have a direct connection
                    already_connected = False
                    for d in env.doors.values():
                        if (d.connects[0] == r1 and d.connects[1] == r2) or (d.connects[0] == r2 and d.connects[1] == r1):
                            already_connected = True
                            break
                    
                    if not already_connected:
                        possible_connections.append((r1, r2))
            
            if possible_connections:
                room1, room2 = random.choice(possible_connections)
                
                description = random.choice(door_descriptions)
                door_descriptions.remove(description)
                if not door_descriptions:
                    door_descriptions = list(door_descriptions)
                
                door = Door(door_id, description, (room1, room2))
                env.add_door(door)
                non_mst_doors.append(door_id)
                door_id += 1
        
        # Now implement a traversal-based locking and key placement strategy
        # We'll first create a traversal order of the MST to place keys and locks
        
        # Start with a random walk from the starting room
        traversed_rooms = {1}  # Start room is already traversed
        frontier_rooms = set()  # Rooms connected to traversed rooms but not yet traversed
        
        # Get initial frontier
        for door_id in mst_doors:
            door = env.doors[door_id]
            if door.connects[0] == 1:
                frontier_rooms.add(door.connects[1])
            elif door.connects[1] == 1:
                frontier_rooms.add(door.connects[0])
        
        # Create a list to track door/key pairs that will ensure solvability
        required_keys = []  # (door_id, key_id) pairs
        
        # Random connected search traversal to place keys and locks
        # This ensures all rooms remain accessible
        key_id = 201
        doors_to_lock = []
        
        # Determine how many MST doors to lock
        mst_lock_count = min(int(len(mst_doors) * lock_percentage), len(mst_doors) - 1)  # Keep at least one MST door unlocked
        mst_doors_to_lock = random.sample(mst_doors[1:], mst_lock_count)  # Skip first door to ensure initial room access
        doors_to_lock.extend(mst_doors_to_lock)
        
        # Traverse the MST
        while frontier_rooms:
            # Pick a random room from the frontier
            next_room = random.choice(list(frontier_rooms))
            frontier_rooms.remove(next_room)
            traversed_rooms.add(next_room)
            
            # If we're locking a door that leads to this room, place its key in an already traversed room
            for door_id in mst_doors_to_lock:
                door = env.doors[door_id]
                if next_room in door.connects and not any(key_pair[0] == door_id for key_pair in required_keys):
                    # This door should be locked and needs a key
                    door.locked = True
                    door.key_id = key_id
                    
                    # Place the key in a previously traversed room (excluding the current room)
                    possible_key_rooms = list(traversed_rooms - {next_room})
                    key_room = random.choice(possible_key_rooms) if possible_key_rooms else 1
                    
                    # Create the key
                    description = random.choice(key_descriptions)
                    key_descriptions.remove(description)
                    if not key_descriptions:
                        key_descriptions = list(key_descriptions)
                    
                    key = Key(key_id, description)
                    env.add_key(key, key_room)
                    
                    required_keys.append((door_id, key_id))
                    key_id += 1
            
            # Update the frontier with any new adjacent rooms
            for door_id in mst_doors:
                door = env.doors[door_id]
                if next_room in door.connects:
                    other_room = door.get_other_room(next_room)
                    if other_room not in traversed_rooms:
                        frontier_rooms.add(other_room)
        
        # Lock some additional non-MST doors
        non_mst_lock_count = min(int(len(non_mst_doors) * lock_percentage * 1.5), len(non_mst_doors))
        non_mst_doors_to_lock = random.sample(non_mst_doors, non_mst_lock_count)
        doors_to_lock.extend(non_mst_doors_to_lock)
        
        # Create keys for the remaining locked doors
        for door_id in non_mst_doors_to_lock:
            door = env.doors[door_id]
            door.locked = True
            door.key_id = key_id
            
            # Place key in a random room
            key_room = random.randint(1, num_rooms)
            
            # Create the key
            if key_descriptions:
                description = random.choice(key_descriptions)
                key_descriptions.remove(description)
            else:
                description = f"A mysterious key with the number {key_id} engraved on it."
            
            key = Key(key_id, description)
            env.add_key(key, key_room)
            
            required_keys.append((door_id, key_id))
            key_id += 1
        
        # Add red herring keys
        for _ in range(red_herring_keys):
            # Place key in a random room
            key_room = random.randint(1, num_rooms)
            
            # Create the key
            if key_descriptions:
                description = random.choice(key_descriptions)
                key_descriptions.remove(description)
            else:
                description = f"A decorative key that doesn't seem to fit any lock you've seen."
            
            key = Key(key_id, description)
            env.add_key(key, key_room)
            key_id += 1
        
        # Set start and target rooms
        env.set_start_room(1)
        env.set_target_room(num_rooms)  # Last room is the target
        
        return env



def visualize_environment(env, filename=None):
    """
    Visualize an environment as a graph.
    
    Args:
        env: The environment to visualize
        filename: Optional filename to save the visualization as an image
    """
    # Create a graph
    G = nx.Graph()
    
    # Add nodes (rooms)
    for room_id, room in env.rooms.items():
        # Get keys in this room
        key_ids = room.keys.copy()  # Use copy to avoid modification issues
        
        # Create label: room_id [key_id1, key_id2,...]
        if key_ids:
            label = f"{room_id} {key_ids}"
        else:
            label = f"{room_id} []"
            
        # Add node with its label
        G.add_node(room_id, label=label)
    
    # Add edges (doors)
    for door_id, door in env.doors.items():
        room1, room2 = door.connects
        
        # Create label: door_id [key_id] or door_id []
        if door.locked and door.key_id is not None:
            label = f"{door_id} [{door.key_id}]"
        else:
            label = f"{door_id} []"
        
        # Determine line style based on whether door is locked
        style = 'dashed' if door.locked else 'solid'
        
        # Add edge with its properties
        G.add_edge(room1, room2, id=door_id, label=label, style=style)
    
    # Create the plot with a larger figure size to accommodate larger text and nodes
    plt.figure(figsize=(16, 12))
    
    # Define node colors
    node_colors = []
    for node in G.nodes():
        if node == env.start_room:
            node_colors.append('red')  # Start room
        elif node == env.target_room:
            node_colors.append('green')  # Target room
        else:
            node_colors.append('skyblue')  # Regular rooms
    
    # Define positions for the nodes
    pos = nx.spring_layout(G, seed=42)  # Use seed for consistent layout
    
    # Draw nodes (rooms) with 1.5x larger size
    nx.draw_networkx_nodes(G, pos, node_size=3000, node_color=node_colors, alpha=0.8)
    
    # Draw edges (doors) with different styles for locked/unlocked
    solid_edges = [(u, v) for u, v, d in G.edges(data=True) if d['style'] == 'solid']
    dashed_edges = [(u, v) for u, v, d in G.edges(data=True) if d['style'] == 'dashed']
    
    nx.draw_networkx_edges(G, pos, edgelist=solid_edges, width=2)
    nx.draw_networkx_edges(G, pos, edgelist=dashed_edges, width=2, style='dashed')
    
    # Draw node labels with 2x larger font
    nx.draw_networkx_labels(G, pos, labels=nx.get_node_attributes(G, 'label'), font_size=20)
    
    # Draw edge labels with 2x larger font
    edge_labels = {(u, v): d['label'] for u, v, d in G.edges(data=True)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=16)
    
    # Set title
    plt.title(f"Environment: {env.name}")
    plt.axis('off')  # Hide axis
    
    # Save to file if filename is provided
    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
    
    # Show the plot
    plt.tight_layout()
    plt.show()
