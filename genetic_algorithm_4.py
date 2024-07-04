import random


# source:  https://www.youtube.com/watch?v=8NrNX_jCkjw&list=PLSM8fkP9ppPruxyt1r1nWJxaxWUX3yabg&index=2


class Data:
    ROOMS = [["R1", 25], ["R2", 45], ["R3", 35]]
    MEETING_TIMES = [
        ["MT1", "MWF 09:00 - 10:00"],
        ["MT2", "MWF 10:00 - 11:00"],
        ["MT3", "TTH 09:00 - 10:30"],
        ["MT4", "TTH 10:30 - 12:00"]

    ]
    INSTRUCTORS = [
        ["11", "Dr Loqman Samani"],
        ["12", "Dr Hossain Khosh"],
        ["13", "Dr Chia Bamshad"],
        ["14", "Dr X X"]
    ]

    def __init__(self):
        self.rooms = []
        self.meeting_times = []
        self.instructors = []

        for i in range(0, len(self.ROOMS)):
            self.rooms.append(Room(self.ROOMS[i][0], self.ROOMS[i][1]))
        for i in range(0, len(self.MEETING_TIMES)):
            self.meeting_times.append(MeetingTime(self.MEETING_TIMES[i][0], self.MEETING_TIMES[i][1]))
        for i in range(0, len(self.INSTRUCTORS)):
            self.instructors.append(Instructor(self.INSTRUCTORS[i][0], self.INSTRUCTORS[i][1]))

        course1 = Course("C1", "325K", [self.instructors[0], self.instructors[1]], 25)
        course2 = Course("C2", "319K", [self.instructors[0], self.instructors[1], self.instructors[2]], 35)
        course3 = Course("C3", "343K", [self.instructors[0], self.instructors[1]], 25)
        course4 = Course("C4", "417K", [self.instructors[2], self.instructors[3]], 30)
        course5 = Course("C5", "311C", [self.instructors[3]], 35)
        course6 = Course("C6", "319K", [self.instructors[0], self.instructors[2]], 45)
        course7 = Course("C7", "303L", [self.instructors[1], self.instructors[3]], 45)

        self.courses = [course1, course2, course3, course4, course5, course6, course7]

        dept1 = Department("MATH", [course1, course2])
        dept2 = Department("EE", [course2, course4, course5])
        dept3 = Department("PHY", [course6, course7])

        self.depts = [dept1, dept2, dept3]
        self.num_classes = 0
        for i in range(0, len(self.depts)):
            self.num_classes += len(self.depts[i].get_courses())

    def get_rooms(self): return self.rooms
    def get_instructors(self): return self.instructors

    def get_courses(self): return self.courses

    def get_depts(self): return self.depts
    def get_num_classes(self): return self.num_classes

    def get_meeting_times(self): return self.meeting_times


class Schedule:
    def __init__(self):
        self.data = data
        self.classes = []
        self.num_conflicts = 0
        self.fitness = -1
        self.class_num = 0
        self.is_fitness_changed = True

    def get_classes(self):
        self.is_fitness_changed = True
        return self.classes

    def get_num_conflicts(self):
        return self.num_conflicts

    def get_fitness(self):
        if self.is_fitness_changed:
            self.fitness = self.calculate_fitness()
            self.is_fitness_changed = False
        return self.fitness

    def calculate_fitness(self):
        self.num_conflicts = 0
        classes = self.get_classes()
        for i in range(0, len(classes)):
            if (classes[i].get_room().get_seatting_capacity() < classes[i].get_courses().get_max_num_studentds()):
                for j in range(0, len(classes)):
                    if (j >= i):
                        if (classes[i].get_meeting_times() == classes[j].get_meeting_times() and classes[i].get_id() != classes[j]):
                            # TODO: till here
                            pass





    def initialize(self):

        depts = self.data.get_depts()
        for i in range(0, len(depts)):
            courses = depts[i].get_courses()
            for j in range(0, len(courses)):
                new_class = Class(self.class_num, depts[i], courses[j])
                self.class_num += 1
                new_class.set_meeting_time(data.get_meeting_times()[random.randrange(0, len(data.get_meeting_times()))])
                new_class.set_room(data.get_rooms()[random.randrange(0, len(data.get_rooms()))])
                new_class.set_instructor(courses[j].get_instructors()[random.randrange(0, len(courses[j].get_instructors()))])
                self.classes.append(new_class)
        return self






class Population:
    pass


class GeneticAlgorithm:
    pass


class Course:
    def __init__(self, number, name, instructors, max_num_student):
        self.number = number
        self.name = name
        self.instructors = instructors
        self.max_num_student = max_num_student

    def get_number(self): return self.number
    def get_name(self): return self.name
    def get_instructors(self): return self.instructors
    def get_max_num_student(self): return self.max_num_student

    def __str__(self): return self.name




class Instructor:
    def __init__(self, id, name):
        self.id = id
        self.name = name

    def get_id(self): return self.id

    def get_name(self): return self.name

    def __str__(self): return self.name





class Room:
    def __init__(self, number, seat_capacity):
        self.number = number
        self.seat_capacity = seat_capacity

    def get_number(self): return self.number
    def get_seat_capacity(self): return self.seat_capacity



class MeetingTime:
    def __init__(self, id, time):
        self.id = id
        self.time = time

    def get_id(self): return self.id

    def get_time(self): return self.time



class Department:
    def __init__(self, name, courses):
        self.name = name
        self.courses = courses

    def get_name(self): return self.name
    def get_courses(self): return self.courses



class Class:
    def __init__(self, id, dept, course):
        self.id = id
        self.dept = dept
        self.course = course
        self.meeting_time = None
        self.room = None
        self.instructor = None

    def get_id(self): return self.id
    def get_dept(self): return self.dept

    def get_course(self): return self.course

    def get_meeting_time(self): return self.meeting_time
    def get_room(self): return self.room

    def get_instructor(self): return self.instructor

    def set_meeting_time(self, meeting_time): self.meeting_time = meeting_time

    def set_instructor(self, instructor):  self.instructor = instructor

    def set_room(self, room): self.room = room

    def __str__(self):

        return f"{self.dept.get_name()}, {self.course.get_number()}, {self.room.get_number()}, {self.instructor.get_id()}, {self.meeting_time.get_id()}"



data = Data()
