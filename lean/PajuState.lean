/-! パジュの状態図の模型（`loop/paju_state_lean.py` がコードから作る。手で直さない）。 -/

set_option linter.unusedVariables false

inductive Ev where
  | «request_opened»
  | «quiet_request_opened»
  | «sleep_began»
  | «speech_began»
  | «request_closed»
  | «speech_ended»
  | «woke»
  | «limit_reached»
  | «window_opened»
  | «tentative_called»
  | «window_lapsed»
  | «tentative_lapsed»
  | «tentative_dropped»
  | «confirm_asked»
  | «offer_made»
  | «heard_asked»
  | «answer_cleared»
  | «music_gated»
  | «timer_gated»
  | «gate_opened»
  | «ring_began»
  | «ring_ended»
  | «song_began»
  | «song_quieted»
  | «song_ended»
  | «song_restored»
  | «silence_began»
  | «silence_ended»
  | «thinking_cue_began»
  | «cue_began»
  | «voice_began»
  | «thinking_cue_ended»
  | «cue_ended»
  | «voice_ended»
  | «neck_began»
  | «neck_ended»
  deriving DecidableEq, Repr

def Ev.all : List Ev := [.«request_opened», .«quiet_request_opened», .«sleep_began», .«speech_began», .«request_closed», .«speech_ended», .«woke», .«limit_reached», .«window_opened», .«tentative_called», .«window_lapsed», .«tentative_lapsed», .«tentative_dropped», .«confirm_asked», .«offer_made», .«heard_asked», .«answer_cleared», .«music_gated», .«timer_gated», .«gate_opened», .«ring_began», .«ring_ended», .«song_began», .«song_quieted», .«song_ended», .«song_restored», .«silence_began», .«silence_ended», .«thinking_cue_began», .«cue_began», .«voice_began», .«thinking_cue_ended», .«cue_ended», .«voice_ended», .«neck_began», .«neck_ended»]

inductive Head where
  | «idle»
  | «thinking»
  | «answering»
  | «sleeping»
  deriving DecidableEq, Repr

def Head.all : List Head := [.«idle», .«thinking», .«answering», .«sleeping»]

def Head.step («is_sleeping» : Bool) («silenced_outside» : Bool) («is_head_idle» : Bool) : Head → Ev → Option Head
  | .«idle», .«request_opened» => some .«thinking»
  | .«idle», .«quiet_request_opened» => some .«thinking»
  | .«idle», .«sleep_began» => some .«sleeping»
  | .«thinking», .«speech_began» => some .«answering»
  | .«thinking», .«request_closed» => some .«idle»
  | .«answering», .«speech_ended» => some .«thinking»
  | .«answering», .«request_closed» => some .«idle»
  | .«sleeping», .«woke» => some .«idle»
  | _, _ => none

/-- head：n 歩以内に初めの状態（idle）へ戻れるか（条件がみな偽のときの移りで見る）。 -/
def Head.back (n : Nat) (s : Head) : Bool :=
  match n with
  | 0 => s == .«idle»
  | n + 1 => s == .«idle» || Ev.all.any (fun e => match Head.step false false false s e with | some t => Head.back n t | none => false)

theorem head_has_no_dead_end : Head.all.all (Head.back 4) = true := by decide

inductive Listening where
  | «hearing»
  | «deaf»
  deriving DecidableEq, Repr

def Listening.all : List Listening := [.«hearing», .«deaf»]

def Listening.step («is_sleeping» : Bool) («silenced_outside» : Bool) («is_head_idle» : Bool) : Listening → Ev → Option Listening
  | .«hearing», .«request_opened» => if !«is_sleeping» then some .«deaf» else none
  | .«hearing», .«speech_began» => some .«deaf»
  | .«deaf», .«speech_ended» => some .«hearing»
  | .«deaf», .«request_closed» => some .«hearing»
  | .«deaf», .«limit_reached» => some .«hearing»
  | _, _ => none

/-- listening：n 歩以内に初めの状態（hearing）へ戻れるか（条件がみな偽のときの移りで見る）。 -/
def Listening.back (n : Nat) (s : Listening) : Bool :=
  match n with
  | 0 => s == .«hearing»
  | n + 1 => s == .«hearing» || Ev.all.any (fun e => match Listening.step false false false s e with | some t => Listening.back n t | none => false)

theorem listening_has_no_dead_end : Listening.all.all (Listening.back 2) = true := by decide

inductive Window where
  | «shut»
  | «open»
  | «tentative»
  deriving DecidableEq, Repr

def Window.all : List Window := [.«shut», .«open», .«tentative»]

def Window.step («is_sleeping» : Bool) («silenced_outside» : Bool) («is_head_idle» : Bool) : Window → Ev → Option Window
  | .«shut», .«window_opened» => some .«open»
  | .«shut», .«tentative_called» => some .«tentative»
  | .«open», .«window_opened» => some .«open»
  | .«open», .«window_lapsed» => some .«shut»
  | .«tentative», .«window_opened» => some .«open»
  | .«tentative», .«tentative_called» => some .«tentative»
  | .«tentative», .«tentative_lapsed» => some .«shut»
  | .«tentative», .«tentative_dropped» => some .«shut»
  | _, _ => none

/-- window：n 歩以内に初めの状態（shut）へ戻れるか（条件がみな偽のときの移りで見る）。 -/
def Window.back (n : Nat) (s : Window) : Bool :=
  match n with
  | 0 => s == .«shut»
  | n + 1 => s == .«shut» || Ev.all.any (fun e => match Window.step false false false s e with | some t => Window.back n t | none => false)

theorem window_has_no_dead_end : Window.all.all (Window.back 3) = true := by decide

inductive Awaited where
  | «no_answer»
  | «confirm_answer»
  | «offer_answer»
  | «heard_answer»
  deriving DecidableEq, Repr

def Awaited.all : List Awaited := [.«no_answer», .«confirm_answer», .«offer_answer», .«heard_answer»]

def Awaited.step («is_sleeping» : Bool) («silenced_outside» : Bool) («is_head_idle» : Bool) : Awaited → Ev → Option Awaited
  | .«no_answer», .«confirm_asked» => some .«confirm_answer»
  | .«no_answer», .«offer_made» => some .«offer_answer»
  | .«no_answer», .«heard_asked» => some .«heard_answer»
  | .«confirm_answer», .«offer_made» => some .«offer_answer»
  | .«confirm_answer», .«heard_asked» => some .«heard_answer»
  | .«confirm_answer», .«answer_cleared» => some .«no_answer»
  | .«offer_answer», .«confirm_asked» => some .«confirm_answer»
  | .«offer_answer», .«heard_asked» => some .«heard_answer»
  | .«offer_answer», .«answer_cleared» => some .«no_answer»
  | .«heard_answer», .«confirm_asked» => some .«confirm_answer»
  | .«heard_answer», .«offer_made» => some .«offer_answer»
  | .«heard_answer», .«answer_cleared» => some .«no_answer»
  | _, _ => none

/-- awaited：n 歩以内に初めの状態（no_answer）へ戻れるか（条件がみな偽のときの移りで見る）。 -/
def Awaited.back (n : Nat) (s : Awaited) : Bool :=
  match n with
  | 0 => s == .«no_answer»
  | n + 1 => s == .«no_answer» || Ev.all.any (fun e => match Awaited.step false false false s e with | some t => Awaited.back n t | none => false)

theorem awaited_has_no_dead_end : Awaited.all.all (Awaited.back 4) = true := by decide

inductive Entrance where
  | «ear_open»
  | «music_only»
  | «timer_only»
  deriving DecidableEq, Repr

def Entrance.all : List Entrance := [.«ear_open», .«music_only», .«timer_only»]

def Entrance.step («is_sleeping» : Bool) («silenced_outside» : Bool) («is_head_idle» : Bool) : Entrance → Ev → Option Entrance
  | .«ear_open», .«music_gated» => some .«music_only»
  | .«ear_open», .«timer_gated» => some .«timer_only»
  | .«music_only», .«gate_opened» => some .«ear_open»
  | .«music_only», .«timer_gated» => some .«timer_only»
  | .«timer_only», .«gate_opened» => some .«ear_open»
  | .«timer_only», .«music_gated» => some .«music_only»
  | _, _ => none

/-- entrance：n 歩以内に初めの状態（ear_open）へ戻れるか（条件がみな偽のときの移りで見る）。 -/
def Entrance.back (n : Nat) (s : Entrance) : Bool :=
  match n with
  | 0 => s == .«ear_open»
  | n + 1 => s == .«ear_open» || Ev.all.any (fun e => match Entrance.step false false false s e with | some t => Entrance.back n t | none => false)

theorem entrance_has_no_dead_end : Entrance.all.all (Entrance.back 3) = true := by decide

inductive Alarm where
  | «alarm_quiet»
  | «alarm_ringing»
  deriving DecidableEq, Repr

def Alarm.all : List Alarm := [.«alarm_quiet», .«alarm_ringing»]

def Alarm.step («is_sleeping» : Bool) («silenced_outside» : Bool) («is_head_idle» : Bool) : Alarm → Ev → Option Alarm
  | .«alarm_quiet», .«ring_began» => some .«alarm_ringing»
  | .«alarm_ringing», .«ring_ended» => some .«alarm_quiet»
  | _, _ => none

/-- alarm：n 歩以内に初めの状態（alarm_quiet）へ戻れるか（条件がみな偽のときの移りで見る）。 -/
def Alarm.back (n : Nat) (s : Alarm) : Bool :=
  match n with
  | 0 => s == .«alarm_quiet»
  | n + 1 => s == .«alarm_quiet» || Ev.all.any (fun e => match Alarm.step false false false s e with | some t => Alarm.back n t | none => false)

theorem alarm_has_no_dead_end : Alarm.all.all (Alarm.back 2) = true := by decide

inductive Song where
  | «song_stopped»
  | «song_playing»
  | «song_ducked»
  deriving DecidableEq, Repr

def Song.all : List Song := [.«song_stopped», .«song_playing», .«song_ducked»]

def Song.step («is_sleeping» : Bool) («silenced_outside» : Bool) («is_head_idle» : Bool) : Song → Ev → Option Song
  | .«song_stopped», .«song_began» => some .«song_playing»
  | .«song_stopped», .«song_quieted» => some .«song_ducked»
  | .«song_playing», .«song_quieted» => some .«song_ducked»
  | .«song_playing», .«song_ended» => some .«song_stopped»
  | .«song_ducked», .«song_restored» => some .«song_playing»
  | .«song_ducked», .«song_ended» => some .«song_stopped»
  | _, _ => none

/-- song：n 歩以内に初めの状態（song_stopped）へ戻れるか（条件がみな偽のときの移りで見る）。 -/
def Song.back (n : Nat) (s : Song) : Bool :=
  match n with
  | 0 => s == .«song_stopped»
  | n + 1 => s == .«song_stopped» || Ev.all.any (fun e => match Song.step false false false s e with | some t => Song.back n t | none => false)

theorem song_has_no_dead_end : Song.all.all (Song.back 3) = true := by decide

inductive Hush where
  | «may_speak»
  | «silent»
  deriving DecidableEq, Repr

def Hush.all : List Hush := [.«may_speak», .«silent»]

def Hush.step («is_sleeping» : Bool) («silenced_outside» : Bool) («is_head_idle» : Bool) : Hush → Ev → Option Hush
  | .«may_speak», .«silence_began» => some .«silent»
  | .«silent», .«silence_began» => some .«silent»
  | .«silent», .«silence_ended» => some .«may_speak»
  | _, _ => none

/-- hush：n 歩以内に初めの状態（may_speak）へ戻れるか（条件がみな偽のときの移りで見る）。 -/
def Hush.back (n : Nat) (s : Hush) : Bool :=
  match n with
  | 0 => s == .«may_speak»
  | n + 1 => s == .«may_speak» || Ev.all.any (fun e => match Hush.step false false false s e with | some t => Hush.back n t | none => false)

theorem hush_has_no_dead_end : Hush.all.all (Hush.back 2) = true := by decide

inductive Voice where
  | «no_voice»
  | «thinking_cue»
  | «cue»
  | «speaking»
  deriving DecidableEq, Repr

def Voice.all : List Voice := [.«no_voice», .«thinking_cue», .«cue», .«speaking»]

def Voice.step («is_sleeping» : Bool) («silenced_outside» : Bool) («is_head_idle» : Bool) : Voice → Ev → Option Voice
  | .«no_voice», .«thinking_cue_began» => some .«thinking_cue»
  | .«no_voice», .«cue_began» => some .«cue»
  | .«no_voice», .«voice_began» => if !«silenced_outside» then some .«speaking» else none
  | .«thinking_cue», .«thinking_cue_ended» => some .«no_voice»
  | .«thinking_cue», .«cue_began» => some .«cue»
  | .«thinking_cue», .«voice_began» => if !«silenced_outside» then some .«speaking» else none
  | .«cue», .«thinking_cue_began» => some .«thinking_cue»
  | .«cue», .«cue_began» => some .«cue»
  | .«cue», .«cue_ended» => some .«no_voice»
  | .«cue», .«voice_began» => if !«silenced_outside» then some .«speaking» else none
  | .«speaking», .«voice_ended» => some .«no_voice»
  | _, _ => none

/-- voice：n 歩以内に初めの状態（no_voice）へ戻れるか（条件がみな偽のときの移りで見る）。 -/
def Voice.back (n : Nat) (s : Voice) : Bool :=
  match n with
  | 0 => s == .«no_voice»
  | n + 1 => s == .«no_voice» || Ev.all.any (fun e => match Voice.step false false false s e with | some t => Voice.back n t | none => false)

theorem voice_has_no_dead_end : Voice.all.all (Voice.back 4) = true := by decide

inductive Body where
  | «body_still»
  | «neck_moving»
  | «patrolling»
  deriving DecidableEq, Repr

def Body.all : List Body := [.«body_still», .«neck_moving», .«patrolling»]

def Body.step («is_sleeping» : Bool) («silenced_outside» : Bool) («is_head_idle» : Bool) : Body → Ev → Option Body
  | .«body_still», .«neck_began» => some .«neck_moving»
  | .«body_still», .«neck_ended» => some .«body_still»
  | .«body_still», .«quiet_request_opened» => if «is_head_idle» then some .«patrolling» else none
  | .«neck_moving», .«neck_began» => some .«neck_moving»
  | .«neck_moving», .«neck_ended» => some .«body_still»
  | .«neck_moving», .«quiet_request_opened» => if «is_head_idle» then some .«patrolling» else none
  | .«patrolling», .«neck_began» => some .«patrolling»
  | .«patrolling», .«neck_ended» => some .«patrolling»
  | .«patrolling», .«request_closed» => some .«body_still»
  | _, _ => none

/-- body：n 歩以内に初めの状態（body_still）へ戻れるか（条件がみな偽のときの移りで見る）。 -/
def Body.back (n : Nat) (s : Body) : Bool :=
  match n with
  | 0 => s == .«body_still»
  | n + 1 => s == .«body_still» || Ev.all.any (fun e => match Body.step false false false s e with | some t => Body.back n t | none => false)

theorem body_has_no_dead_end : Body.all.all (Body.back 3) = true := by decide

/-- 頭が寝ているとき、`request_opened` では頭も耳の考えているあいだも移らない（決定ク・ダ）。 -/
theorem sleeping_takes_no_request :
    Head.all.all (fun h => Listening.all.all (fun l =>
      h != .«sleeping» ||
        (Head.step (h == .«sleeping») false false h .«request_opened» == none &&
         Listening.step (h == .«sleeping») false false l .«request_opened» == none))) = true := by
  decide

/-- 新しい問いは、どの状態からも前の問いを上書きし、その問いの答えだけを待つ（決定ノ）。 -/
theorem a_new_question_replaces_the_old :
    Awaited.all.all (fun a =>
      (Awaited.step false false false a .«confirm_asked» == some .«confirm_answer» || a == .«confirm_answer») &&
      (Awaited.step false false false a .«offer_made» == some .«offer_answer» || a == .«offer_answer») &&
      (Awaited.step false false false a .«heard_asked» == some .«heard_answer» || a == .«heard_answer»)) = true := by
  decide

/-- 黙る依頼が効いていて求めの外の声なら、声は話しているへ移らない。 -/
theorem silenced_takes_no_outside_voice :
    Voice.all.all (fun v => Voice.step false true false v .«voice_began» == none) = true := by
  decide
